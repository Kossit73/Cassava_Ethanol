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
