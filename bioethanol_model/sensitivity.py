from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .financial_model import CassavaBioethanolModel


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
    parameter_std: Dict[str, float],
    iterations: int = 1000,
    random_seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    table = model.input_page.global_inputs
    if table.placeholder:
        return pd.DataFrame()
    params = table.data.set_index("Parameter")["Value"].to_dict()
    rows = []
    for _ in range(iterations):
        for param, std in parameter_std.items():
            if param in params:
                sampled = rng.normal(params[param], std)
                table.data.loc[table.data["Parameter"] == param, "Value"] = sampled
        result = model.build()
        rows.append(
            {
                "Project NPV": result["metrics"]["Project NPV"],
                "Project IRR": result["metrics"]["Project IRR"],
                "Equity IRR": result["metrics"]["Equity IRR"],
            }
        )
    for param, value in params.items():
        table.data.loc[table.data["Parameter"] == param, "Value"] = value
    return pd.DataFrame(rows)


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
