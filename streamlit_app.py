"""Streamlit interface for the cassava ethanol financial model."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st
import yaml

from cassava_ethanol import (
    CassavaEthanolModel,
    Scenario,
    ScenarioRunner,
    format_currency,
    format_percentage,
)
from cassava_ethanol.inputs import (
    CapitalPlan,
    FeedstockAssumptions,
    FinancialAssumptions,
    ModelInputs,
    OperatingCosts,
    PlantProfile,
    ProductPricing,
    iter_dotted_paths,
)

EXAMPLE_CONFIG_PATH = Path(__file__).parent / "examples" / "base_config.yaml"
EXAMPLE_SCENARIOS_PATH = Path(__file__).parent / "examples" / "sample_scenarios.yaml"


def _load_yaml_from_uploader(upload) -> str | None:
    if upload is None:
        return None
    if hasattr(upload, "getvalue"):
        return upload.getvalue().decode("utf-8")
    return None


def _load_model_inputs(raw: Dict[str, Dict[str, float]]) -> ModelInputs:
    try:
        plant = PlantProfile(**raw["plant"])
        feedstock = FeedstockAssumptions(**raw["feedstock"])
        pricing = ProductPricing(**raw["pricing"])
        operating = OperatingCosts(**raw["operating_costs"])
        capital = CapitalPlan(**raw["capital"])
        financial = FinancialAssumptions(**raw["financial"])
    except KeyError as exc:  # pragma: no cover - user input guard
        raise ValueError(f"Missing configuration section: {exc.args[0]}") from exc
    return ModelInputs(
        plant=plant,
        feedstock=feedstock,
        pricing=pricing,
        operating_costs=operating,
        capital=capital,
        financial=financial,
    )


def _list_numeric_override_fields(inputs: ModelInputs) -> List[Tuple[str, float]]:
    options: List[Tuple[str, float]] = []
    for path in iter_dotted_paths(inputs):
        if path.count(".") != 1:
            continue
        section, field = path.split(".")
        value = getattr(getattr(inputs, section), field)
        if isinstance(value, (int, float)):
            options.append((path, float(value)))
    return sorted(options)


def _apply_overrides(inputs: ModelInputs, overrides: Dict[str, float]) -> ModelInputs:
    if not overrides:
        return inputs
    return inputs.copy_with_overrides(overrides)


def _cash_flow_records(result) -> List[Dict[str, float]]:
    rows = []
    for row in result.cash_flows:
        rows.append(
            {
                "Year": row.year,
                "Production (L)": row.production_liters,
                "Revenue": row.revenue,
                "Feedstock": row.feedstock_cost,
                "Variable Opex": row.variable_operating_cost,
                "Fixed Opex": row.fixed_operating_cost,
                "Maintenance": row.maintenance_cost,
                "EBITDA": row.ebitda,
                "Free Cash Flow": row.free_cash_flow,
                "Discounted Cash Flow": row.discounted_cash_flow,
                "Cumulative": row.cumulative_cash_flow,
            }
        )
    return rows


def _scenario_table(results) -> List[Dict[str, float]]:
    table = []
    for scenario_result in results:
        metrics = scenario_result.results
        table.append(
            {
                "Scenario": scenario_result.scenario.name,
                "NPV": metrics.npv,
                "IRR": metrics.irr,
                "Payback": metrics.payback_year if metrics.payback_year is not None else float("nan"),
            }
        )
    return table


def _load_example_text(path: Path) -> str:
    try:
        return path.read_text()
    except FileNotFoundError:  # pragma: no cover - defensive
        return ""


def _format_metric(label: str, value: float, formatter) -> None:
    if value != value:  # NaN check
        st.metric(label, "n/a")
    else:
        st.metric(label, formatter(value))


def main() -> None:
    st.set_page_config(page_title="Cassava Ethanol Model", layout="wide")
    st.title("Cassava Ethanol Financial Model")
    st.caption(
        "Interactively explore the cassava-based ethanol project financial model, "
        "adjust key assumptions, and compare scenarios."
    )

    default_config_text = _load_example_text(EXAMPLE_CONFIG_PATH)
    config_upload = st.sidebar.file_uploader("Configuration (YAML)", type=["yaml", "yml"])
    config_text = _load_yaml_from_uploader(config_upload) or default_config_text

    st.sidebar.download_button(
        "Download example config",
        data=default_config_text,
        file_name="cassava_base_config.yaml",
    )

    scenario_upload = st.sidebar.file_uploader("Scenario overrides (YAML)", type=["yaml", "yml"], key="scenario")
    default_scenario_text = _load_example_text(EXAMPLE_SCENARIOS_PATH)
    scenario_text = _load_yaml_from_uploader(scenario_upload) or default_scenario_text

    st.sidebar.download_button(
        "Download example scenarios",
        data=default_scenario_text,
        file_name="cassava_scenarios.yaml",
        key="scenario_download",
    )

    st.subheader("Configuration")
    config_text = st.text_area("Model inputs", value=config_text, height=340)

    try:
        config_data = yaml.safe_load(config_text) or {}
        base_inputs = _load_model_inputs(config_data)
    except Exception as exc:  # pragma: no cover - user input guard
        st.error(f"Unable to load configuration: {exc}")
        st.stop()

    overrides_state = st.session_state.setdefault("overrides", {})
    numeric_options = _list_numeric_override_fields(base_inputs)

    with st.sidebar.expander("Ad-hoc overrides", expanded=False):
        if numeric_options:
            selected_label = st.selectbox(
                "Select assumption",
                [label for label, _ in numeric_options],
                key="override_field",
            )
            current_value = next(value for label, value in numeric_options if label == selected_label)
            new_value = st.number_input(
                "Override value",
                value=float(current_value),
                key="override_value",
            )
            if st.button("Apply override", key="apply_override"):
                overrides_state[selected_label] = float(new_value)
        else:  # pragma: no cover - defensive
            st.info("No numeric assumptions available for overrides.")

        if overrides_state:
            st.write("Active overrides:")
            st.json(overrides_state)
            if st.button("Clear overrides", key="clear_overrides"):
                overrides_state.clear()

    override_inputs = _apply_overrides(base_inputs, overrides_state)
    model = CassavaEthanolModel(base_inputs)
    base_results = model.run()

    custom_results = None
    if overrides_state:
        custom_results = CassavaEthanolModel(override_inputs).run()

    st.subheader("Base case results")
    base_cols = st.columns(3)
    with base_cols[0]:
        _format_metric("NPV", base_results.npv, format_currency)
    with base_cols[1]:
        _format_metric("IRR", base_results.irr, format_percentage)
    with base_cols[2]:
        payback = base_results.payback_year if base_results.payback_year is not None else float("nan")
        _format_metric("Payback", payback, lambda v: f"{v:.1f} years")

    st.dataframe(_cash_flow_records(base_results), use_container_width=True)

    if custom_results:
        st.subheader("Overrides scenario")
        override_cols = st.columns(3)
        with override_cols[0]:
            _format_metric("NPV", custom_results.npv, format_currency)
        with override_cols[1]:
            _format_metric("IRR", custom_results.irr, format_percentage)
        with override_cols[2]:
            payback = (
                custom_results.payback_year
                if custom_results.payback_year is not None
                else float("nan")
            )
            _format_metric("Payback", payback, lambda v: f"{v:.1f} years")
        st.dataframe(_cash_flow_records(custom_results), use_container_width=True)

    st.subheader("Scenario comparison")
    st.caption(
        "Provide scenario overrides as a YAML list of {name, overrides}. "
        "Overrides should use the section.field keys shown in the CLI documentation."
    )
    scenario_text = st.text_area("Scenario definitions", value=scenario_text, height=200)

    scenario_results = []
    if scenario_text.strip():
        try:
            raw_scenarios = yaml.safe_load(scenario_text) or []
            scenario_objects = [
                Scenario(name=item["name"], overrides=item["overrides"])
                for item in raw_scenarios
            ]
            runner = ScenarioRunner(base_inputs, scenario_objects)
            scenario_results = runner.run()
        except Exception as exc:  # pragma: no cover - user input guard
            st.error(f"Unable to load scenarios: {exc}")
            scenario_results = []

    if scenario_results:
        st.dataframe(_scenario_table(scenario_results), use_container_width=True)
    else:
        st.info("No scenario results to display.")

    st.subheader("Input snapshot")
    st.json(asdict(override_inputs))


if __name__ == "__main__":
    main()
