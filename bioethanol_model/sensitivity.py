from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .financial_model import CassavaBioethanolModel


@dataclass(frozen=True)
class _DistributionSpec:
    """Describe a supported probability distribution for Monte Carlo sampling."""

    dist: Any
    shape_params: Tuple[str, ...] = ()
    keyword_params: Tuple[str, ...] = ()
    force_size_one: bool = False

    def sample(
        self,
        rng: np.random.Generator,
        raw_values: Mapping[str, Any],
        base_value: float | None = None,
    ) -> float:
        """Return a scalar sample drawn from the distribution.

        Parameters in *raw_values* are coerced to floats (or lists for ``pvals``)
        and supplied to the SciPy distribution.  ``loc`` defaults to the
        provided *base_value* when omitted to keep behaviour aligned with the
        deterministic base inputs.
        """

        shape_args: List[float] = []
        kwargs: Dict[str, Any] = {}

        for key in self.shape_params:
            value = _resolve_parameter_value(key, raw_values.get(key), base_value)
            if value is None:
                raise ValueError(f"Missing required parameter '{key}' for distribution")
            if isinstance(value, (list, tuple, np.ndarray)):
                shape_args.extend([float(v) for v in value])
            else:
                shape_args.append(float(value))

        for key in self.keyword_params:
            value = _resolve_parameter_value(key, raw_values.get(key), base_value)
            if value is None:
                continue
            kwargs[key] = value

        if self.force_size_one and "size" not in kwargs:
            kwargs["size"] = 1

        sample = self.dist.rvs(*shape_args, random_state=rng, **kwargs)
        array = np.asarray(sample)
        if array.size == 0:
            raise ValueError("Distribution returned an empty sample")
        flattened = array.reshape(-1)
        return float(flattened[0])


def _resolve_parameter_value(
    key: str, value: Any, base_value: float | None = None
) -> Any:
    """Normalise editor-provided distribution parameters for SciPy calls."""

    if value is None:
        if key == "loc" and base_value is not None:
            return float(base_value)
        return None

    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            if key == "loc" and base_value is not None:
                return float(base_value)
            return None
        if key == "pvals":
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                parsed = [item.strip() for item in cleaned.split(",") if item.strip()]
            if isinstance(parsed, (list, tuple)):
                return [float(v) for v in parsed]
            raise ValueError("pvals must be a sequence of probabilities")
        try:
            return float(cleaned)
        except ValueError as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"Invalid numeric value for parameter '{key}'") from exc

    if isinstance(value, (int, float, np.number)):
        return float(value)

    if key == "pvals" and isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return [float(v) for v in value]

    return value


MONTE_CARLO_DISTRIBUTIONS: Dict[str, _DistributionSpec] = {
    "Normal": _DistributionSpec(stats.norm, keyword_params=("loc", "scale")),
    "Lognormal": _DistributionSpec(stats.lognorm, shape_params=("s",), keyword_params=("loc", "scale")),
    "Uniform": _DistributionSpec(stats.uniform, keyword_params=("loc", "scale")),
    "Exponential": _DistributionSpec(stats.expon, keyword_params=("loc", "scale")),
    "Binomial": _DistributionSpec(stats.binom, shape_params=("n", "p"), keyword_params=("loc",)),
    "Poisson": _DistributionSpec(stats.poisson, shape_params=("mu",), keyword_params=("loc",)),
    "Geometric": _DistributionSpec(stats.geom, shape_params=("p",), keyword_params=("loc",)),
    "Bernoulli": _DistributionSpec(stats.bernoulli, shape_params=("p",), keyword_params=("loc",)),
    "Chi-Squared": _DistributionSpec(stats.chi2, shape_params=("df",), keyword_params=("loc", "scale")),
    "Gamma": _DistributionSpec(stats.gamma, shape_params=("a",), keyword_params=("loc", "scale")),
    "Weibull (Min)": _DistributionSpec(
        stats.weibull_min, shape_params=("c",), keyword_params=("loc", "scale")
    ),
    "Hypergeometric": _DistributionSpec(
        stats.hypergeom, shape_params=("M", "n", "N"), keyword_params=("loc",)
    ),
    "Multinomial": _DistributionSpec(
        stats.multinomial,
        shape_params=("n",),
        keyword_params=("pvals",),
        force_size_one=True,
    ),
    "Beta": _DistributionSpec(stats.beta, shape_params=("a", "b"), keyword_params=("loc", "scale")),
    "F": _DistributionSpec(stats.f, shape_params=("dfn", "dfd"), keyword_params=("loc", "scale")),
}

MONTE_CARLO_PARAMETER_COLUMNS: Tuple[str, ...] = (
    "Parameter",
    "Distribution",
    "loc",
    "scale",
    "s",
    "n",
    "p",
    "mu",
    "df",
    "a",
    "b",
    "c",
    "M",
    "N",
    "pvals",
    "dfn",
    "dfd",
)

DEFAULT_MONTE_CARLO_ITERATIONS = 250
DEFAULT_MONTE_CARLO_SEED = 42


def available_monte_carlo_distributions() -> List[str]:
    """Return the ordered list of supported Monte Carlo distributions."""

    return list(MONTE_CARLO_DISTRIBUTIONS.keys())


def default_monte_carlo_parameters() -> pd.DataFrame:
    """Seed Monte Carlo configuration with the standard project parameters."""

    data = [
        {
            "Parameter": "Corporate tax rate",
            "Distribution": "Normal",
            "scale": 0.01,
        },
        {
            "Parameter": "Investor share capital",
            "Distribution": "Normal",
            "scale": 0.02,
        },
    ]

    df = pd.DataFrame(data)
    for column in MONTE_CARLO_PARAMETER_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan
    return df[list(MONTE_CARLO_PARAMETER_COLUMNS)]


@dataclass
class SensitivityScenario:
    name: str
    parameter: str
    delta: float


def run_sensitivity(model: CassavaBioethanolModel, scenarios: Iterable[SensitivityScenario]) -> pd.DataFrame:
    base_results = model.build()
    base_metric = base_results["metrics"]["Project NPV"]
    rows = []
    for scenario in scenarios:
        table = model.input_page.global_inputs
        if table.placeholder:
            continue
        if scenario.parameter not in table.data["Parameter"].values:
            continue
        original = table.data.set_index("Parameter").loc[scenario.parameter, "Value"]
        table.data.loc[table.data["Parameter"] == scenario.parameter, "Value"] = original + scenario.delta
        result = model.build()
        rows.append(
            {
                "Scenario": scenario.name,
                "Parameter": scenario.parameter,
                "Delta": scenario.delta,
                "Project NPV": result["metrics"]["Project NPV"],
                "Change vs Base": result["metrics"]["Project NPV"] - base_metric,
            }
        )
        table.data.loc[table.data["Parameter"] == scenario.parameter, "Value"] = original
    return pd.DataFrame(rows)
def monte_carlo_simulation(
    model: CassavaBioethanolModel,
    parameter_configs: Sequence[Mapping[str, Any]] | pd.DataFrame,
    iterations: int = DEFAULT_MONTE_CARLO_ITERATIONS,
    random_seed: int = DEFAULT_MONTE_CARLO_SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    table = model.input_page.global_inputs
    if table.placeholder:
        return pd.DataFrame()

    base_values = table.data.set_index("Parameter")["Value"].to_dict()
    config_records = _normalise_parameter_configs(parameter_configs)
    if not config_records:
        return pd.DataFrame()

    results: List[Dict[str, Any]] = []

    for _ in range(int(iterations)):
        _reset_global_inputs(table.data, base_values)
        for record in config_records:
            param = record["Parameter"]
            spec = MONTE_CARLO_DISTRIBUTIONS.get(record["Distribution"])
            if spec is None:
                continue
            if param not in base_values:
                continue

            base_value = _coerce_float(base_values[param])
            if base_value is None:
                continue

            try:
                sampled = spec.sample(rng, record, base_value=base_value)
            except ValueError:
                continue

            table.data.loc[table.data["Parameter"] == param, "Value"] = sampled

        result = model.build()
        metrics = result.get("metrics", {})
        results.append(
            {
                "Project NPV": metrics.get("Project NPV"),
                "Project IRR": metrics.get("Project IRR"),
                "Equity IRR": metrics.get("Equity IRR"),
            }
        )

    _reset_global_inputs(table.data, base_values)
    return pd.DataFrame(results)


def _normalise_parameter_configs(
    parameter_configs: Sequence[Mapping[str, Any]] | pd.DataFrame,
) -> List[Dict[str, Any]]:
    if isinstance(parameter_configs, pd.DataFrame):
        frame = parameter_configs.replace("", np.nan)
        records = frame.to_dict("records")
    else:
        records = list(parameter_configs)

    normalised: List[Dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        parameter = record.get("Parameter") or record.get("parameter")
        distribution = record.get("Distribution") or record.get("distribution")
        if not parameter or not distribution:
            continue
        distribution_name = str(distribution)
        if distribution_name not in MONTE_CARLO_DISTRIBUTIONS:
            continue
        entry = dict(record)
        entry["Parameter"] = str(parameter)
        entry["Distribution"] = distribution_name
        normalised.append(entry)
    return normalised


def _reset_global_inputs(df: pd.DataFrame, values: Mapping[str, Any]) -> None:
    for parameter, base_value in values.items():
        df.loc[df["Parameter"] == parameter, "Value"] = base_value


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive guard
        return None


def tornado_chart_inputs(
    model: CassavaBioethanolModel,
    drivers: List[Tuple[str, float]],
    scale: float = 0.1,
) -> pd.DataFrame:
    rows = []
    base = model.build()["metrics"]["Project NPV"]
    for param, pct in drivers:
        table = model.input_page.global_inputs
        if table.placeholder:
            continue
        if param not in table.data["Parameter"].values:
            continue
        base_value = table.data.set_index("Parameter").loc[param, "Value"]
        for direction in (-1, 1):
            table.data.loc[table.data["Parameter"] == param, "Value"] = base_value * (1 + direction * scale * pct)
            result = model.build()
            rows.append({"Parameter": param, "Direction": "Down" if direction == -1 else "Up", "NPV": result["metrics"]["Project NPV"]})
        table.data.loc[table.data["Parameter"] == param, "Value"] = base_value
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["Parameter", "Down", "Up", "Impact", "Base"])

    pivot = df.pivot(index="Parameter", columns="Direction", values="NPV")
    for column in ("Down", "Up"):
        if column not in pivot.columns:
            pivot[column] = pd.NA

    pivot["Impact"] = pivot["Up"] - pivot["Down"]
    pivot["Base"] = base
    return pivot.reset_index()
