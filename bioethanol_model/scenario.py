from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import pandas as pd

from .financial_model import CassavaBioethanolModel
from .utils import GoalSeekResult, goal_seek


@dataclass
class ScenarioConfig:
    name: str
    overrides: Dict[str, float]


def apply_scenario(model: CassavaBioethanolModel, config: ScenarioConfig) -> Dict[str, object]:
    original = model.input_page.global_inputs.data.set_index("Parameter")["Value"].to_dict()
    for key, value in config.overrides.items():
        if key in model.input_page.global_inputs.data["Parameter"].values:
            model.input_page.global_inputs.data.loc[
                model.input_page.global_inputs.data["Parameter"] == key, "Value"
            ] = value
    results = model.build()
    for key, value in original.items():
        model.input_page.global_inputs.data.loc[model.input_page.global_inputs.data["Parameter"] == key, "Value"] = value
    return results


def goal_seek_to_target(
    model: CassavaBioethanolModel,
    parameter: str,
    target_metric: str,
    target_value: float,
) -> GoalSeekResult:
    table = model.input_page.global_inputs.data
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
    return outcome


def scenario_comparison(model: CassavaBioethanolModel, configs: Iterable[ScenarioConfig]) -> pd.DataFrame:
    rows = []
    for config in configs:
        result = apply_scenario(model, config)
        row = {"Scenario": config.name}
        row.update(result["metrics"])
        rows.append(row)
    return pd.DataFrame(rows)
