from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from .financial_model import CassavaBioethanolModel
from .inputs import InputLandingPage
from .utils import GoalSeekResult, goal_seek
from .sensitivity import (
    MONTE_CARLO_PARAMETER_ADAPTERS,
    MonteCarloParameterState,
    SCENARIO_PARAMETER_NAMES,
)


@dataclass
class ScenarioConfig:
    name: str
    overrides: Dict[str, float]


def _capture_adapter_states(
    page: InputLandingPage, overrides: Dict[str, float]
) -> Dict[str, MonteCarloParameterState]:
    states: Dict[str, MonteCarloParameterState] = {}
    for parameter in overrides:
        adapter = MONTE_CARLO_PARAMETER_ADAPTERS.get(parameter)
        if adapter is None or parameter in states:
            continue
        try:
            states[parameter] = adapter.capture(page)
        except AttributeError:
            continue
    return states


def apply_scenario(model: CassavaBioethanolModel, config: ScenarioConfig) -> Dict[str, object]:
    table = model.input_page.global_inputs
    base_values: Dict[str, object] = {}

    if {"Parameter", "Value"}.issubset(table.data.columns):
        lookup = table.data.set_index("Parameter")["Value"]
    else:
        lookup = pd.Series(dtype=object)

    adapter_states = _capture_adapter_states(model.input_page, config.overrides)

    for key, value in config.overrides.items():
        if key in lookup.index:
            if key not in base_values:
                base_values[key] = lookup.loc[key]
            table.data.loc[table.data["Parameter"] == key, "Value"] = value
            continue

        adapter = MONTE_CARLO_PARAMETER_ADAPTERS.get(key)
        state = adapter_states.get(key)
        if adapter is None or state is None:
            continue
        try:
            target = float(value)
        except (TypeError, ValueError):
            continue
        adapter.apply(model.input_page, target, state)

    results = model.build()

    for key, original in base_values.items():
        table.data.loc[table.data["Parameter"] == key, "Value"] = original

    for key, state in adapter_states.items():
        adapter = MONTE_CARLO_PARAMETER_ADAPTERS.get(key)
        if adapter is not None:
            adapter.apply(model.input_page, state.base_value, state)

    return results


def goal_seek_to_target(
    model: CassavaBioethanolModel,
    parameter: str,
    target_metric: str,
    target_value: float,
) -> GoalSeekResult:
    table_obj = model.input_page.global_inputs
    if table_obj.placeholder:
        raise ValueError("Global inputs must be provided before running goal seek")

    table = table_obj.data
    if parameter not in table["Parameter"].values:
        raise KeyError(f"Parameter {parameter} not in global inputs")

    base_value = float(table.set_index("Parameter").loc[parameter, "Value"])

    def objective(x: float) -> float:
        table.loc[table["Parameter"] == parameter, "Value"] = x
        result = model.build()
        table.loc[table["Parameter"] == parameter, "Value"] = base_value
        return result["metrics"][target_metric]

    outcome = goal_seek(objective, target_value, base_value)
    table.loc[table["Parameter"] == parameter, "Value"] = base_value
    outcome.target_name = parameter
    return outcome


def scenario_comparison(model: CassavaBioethanolModel, configs: Iterable[ScenarioConfig]) -> pd.DataFrame:
    rows = []
    for config in configs:
        result = apply_scenario(model, config)
        row = {"Scenario": config.name}
        row.update(result["metrics"])
        rows.append(row)
    return pd.DataFrame(rows)


def scenario_parameter_catalog(page: InputLandingPage) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    global_table = page.global_inputs.data if hasattr(page.global_inputs, "data") else pd.DataFrame()
    if not global_table.empty and {"Parameter", "Value"}.issubset(global_table.columns):
        value_map = global_table.set_index("Parameter")["Value"].to_dict()
        unit_map = (
            global_table.set_index("Parameter")["Units"].to_dict()
            if "Units" in global_table.columns
            else {}
        )
    else:
        value_map = {}
        unit_map = {}

    for parameter in SCENARIO_PARAMETER_NAMES:
        if parameter in value_map:
            raw_value = value_map[parameter]
            numeric = pd.to_numeric(pd.Series([raw_value]), errors="coerce").iloc[0]
            rows.append(
                {
                    "Parameter": parameter,
                    "Base Value": float(numeric) if pd.notna(numeric) else np.nan,
                    "Units": unit_map.get(parameter, ""),
                    "Source": "Global Inputs",
                }
            )
            continue

        adapter = MONTE_CARLO_PARAMETER_ADAPTERS.get(parameter)
        if adapter is None:
            continue
        try:
            state = adapter.capture(page)
        except AttributeError:
            continue
        rows.append(
            {
                "Parameter": parameter,
                "Base Value": float(state.base_value) if np.isfinite(state.base_value) else np.nan,
                "Units": adapter.units,
                "Source": adapter.table_attr.replace("_", " ").title(),
            }
        )

    return pd.DataFrame(rows)


def credit_committee_scenario_configs(page: InputLandingPage) -> List[ScenarioConfig]:
    """Return pre-baked credit committee scenarios with correlated stresses."""

    catalog = scenario_parameter_catalog(page)
    if catalog.empty:
        return [ScenarioConfig("Base", {})]
    base_map = {
        str(row["Parameter"]): float(row["Base Value"])
        for _, row in catalog.iterrows()
        if pd.notna(row.get("Base Value"))
    }

    def _scaled(parameter: str, factor: float, fallback: float = 0.0) -> float:
        return float(base_map.get(parameter, fallback) * factor)

    # Correlated stress drivers: lower ethanol price proxy + higher feedstock + ramp delay proxy.
    return [
        ScenarioConfig("Base", {}),
        ScenarioConfig(
            "Downside",
            {
                "Revenue Inputs": _scaled("Revenue Inputs", 0.92, 1.0),
                "Cassava feedstock": _scaled("Cassava feedstock", 1.12, 1.0),
                "Production monthly": _scaled("Production monthly", 0.94, 1.0),
            },
        ),
        ScenarioConfig(
            "Severe Downside",
            {
                "Revenue Inputs": _scaled("Revenue Inputs", 0.82, 1.0),
                "Cassava feedstock": _scaled("Cassava feedstock", 1.30, 1.0),
                "Production monthly": _scaled("Production monthly", 0.85, 1.0),
            },
        ),
        ScenarioConfig(
            "Upside",
            {
                "Revenue Inputs": _scaled("Revenue Inputs", 1.08, 1.0),
                "Cassava feedstock": _scaled("Cassava feedstock", 0.93, 1.0),
                "Production monthly": _scaled("Production monthly", 1.06, 1.0),
            },
        ),
    ]


def reverse_stress_test(
    model: CassavaBioethanolModel,
    *,
    dscr_floor: float = 1.0,
    npv_floor: float = 0.0,
) -> pd.DataFrame:
    """Find correlated stress combinations that break DSCR covenant or NPV > 0."""

    catalog = scenario_parameter_catalog(model.input_page)
    if catalog.empty:
        return pd.DataFrame()

    base_map = {
        str(row["Parameter"]): float(row["Base Value"])
        for _, row in catalog.iterrows()
        if pd.notna(row.get("Base Value"))
    }
    required = ["Revenue Inputs", "Cassava feedstock", "Production monthly"]
    if any(p not in base_map for p in required):
        return pd.DataFrame()

    price_factors = np.linspace(1.0, 0.65, 8)
    feed_factors = np.linspace(1.0, 1.45, 10)
    ramp_factors = np.linspace(1.0, 0.75, 6)

    breaches: Dict[str, Dict[str, float]] = {}

    for pf in price_factors:
        for ff in feed_factors:
            for rf in ramp_factors:
                config = ScenarioConfig(
                    name="Reverse Stress Candidate",
                    overrides={
                        "Revenue Inputs": base_map["Revenue Inputs"] * float(pf),
                        "Cassava feedstock": base_map["Cassava feedstock"] * float(ff),
                        "Production monthly": base_map["Production monthly"] * float(rf),
                    },
                )
                result = apply_scenario(model, config)
                metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
                dscr_min = float(pd.to_numeric(pd.Series([metrics.get("DSCR (min)")]), errors="coerce").iloc[0])
                npv_value = float(pd.to_numeric(pd.Series([metrics.get("Project NPV")]), errors="coerce").iloc[0])

                if "DSCR Covenant Breach" not in breaches and np.isfinite(dscr_min) and dscr_min < dscr_floor:
                    breaches["DSCR Covenant Breach"] = {
                        "Condition": f"DSCR (min) < {dscr_floor:.2f}",
                        "Price Factor": float(pf),
                        "Feedstock Factor": float(ff),
                        "Ramp Factor": float(rf),
                        "DSCR (min)": dscr_min,
                        "Project NPV": npv_value,
                    }

                if "NPV Breach" not in breaches and np.isfinite(npv_value) and npv_value < npv_floor:
                    breaches["NPV Breach"] = {
                        "Condition": f"Project NPV < {npv_floor:,.0f}",
                        "Price Factor": float(pf),
                        "Feedstock Factor": float(ff),
                        "Ramp Factor": float(rf),
                        "DSCR (min)": dscr_min,
                        "Project NPV": npv_value,
                    }

                if len(breaches) >= 2:
                    return pd.DataFrame(list(breaches.values()))

    return pd.DataFrame(list(breaches.values()))
