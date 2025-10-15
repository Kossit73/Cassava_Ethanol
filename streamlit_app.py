"""Streamlit dashboard for the Cassava bioethanol financial model."""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from bioethanol_model import CassavaBioethanolModel
from bioethanol_model.exporter import export_to_excel
from bioethanol_model.inputs import EditableTable, InputLandingPage, default_input_page
from bioethanol_model.scenario import ScenarioConfig, goal_seek_to_target, scenario_comparison
from bioethanol_model.sensitivity import (
    SensitivityScenario,
    monte_carlo_simulation,
    run_sensitivity,
    tornado_chart_inputs,
)


st.set_page_config(page_title="Cassava Bioethanol Model", layout="wide")

DEFAULT_SENSITIVITY_SCENARIOS: List[SensitivityScenario] = [
    SensitivityScenario("Corporate tax +1pp", "Corporate tax rate", 0.01),
    SensitivityScenario("Corporate tax -1pp", "Corporate tax rate", -0.01),
    SensitivityScenario("Discount rate +1pp", "Discount rate", 0.01),
    SensitivityScenario("Discount rate -1pp", "Discount rate", -0.01),
]

DEFAULT_SCENARIO_CONFIGS: List[ScenarioConfig] = [
    ScenarioConfig("Higher investor share", {"Investor share capital": 0.55}),
    ScenarioConfig("Lower investor share", {"Investor share capital": 0.35}),
    ScenarioConfig("Lower tax", {"Corporate tax rate": 0.24}),
]

TORNADO_DRIVERS: List[Tuple[str, float]] = [
    ("Corporate tax rate", 1.0),
    ("Investor share capital", 1.0),
    ("Owner share capital", 1.0),
    ("Discount rate", 1.0),
]

MONTE_CARLO_STD = {"Corporate tax rate": 0.01, "Investor share capital": 0.02}
MONTE_CARLO_ITERATIONS = 250
MONTE_CARLO_SEED = 42

def _load_session_inputs() -> InputLandingPage:
    """Return the mutable input landing page stored in session state."""
    if "input_page" not in st.session_state:
        st.session_state.input_page = default_input_page()
    return st.session_state.input_page


def _build_model_snapshot(page: InputLandingPage) -> tuple[CassavaBioethanolModel, Dict[str, object]]:
    """Create a model/result pair from a deep copy of the landing-page inputs."""

    snapshot = copy.deepcopy(page)
    model = CassavaBioethanolModel(snapshot)
    return model, model.build()


def _generate_excel_bytes(model: CassavaBioethanolModel) -> bytes:
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir) / "Cassava_Bioethanol_Financial_Model.xlsx"
        export_to_excel(model, temp_path)
        return temp_path.read_bytes()


def _update_projection(page: InputLandingPage) -> None:
    """Render projection horizon controls within the main layout."""

    st.subheader("Projection Horizon")
    start_col, end_col = st.columns(2)
    start = start_col.number_input(
        "Start Year",
        min_value=2000,
        max_value=2100,
        value=int(page.projection.start_year),
        step=1,
        key="projection_start_year",
    )
    end = end_col.number_input(
        "End Year",
        min_value=start,
        max_value=2125,
        value=int(page.projection.end_year),
        step=1,
        key="projection_end_year",
    )
    page.projection.start_year = int(start)
    page.projection.end_year = int(end)


def _key_assumptions_controls(table: EditableTable) -> None:
    """Expose frequently tweaked assumptions inside the main page."""

    st.subheader("Key Assumptions")
    df = table.data.copy()
    slider_cfg = {
        "Corporate tax rate": dict(min_value=0.0, max_value=0.7, step=0.01),
        "Investor share capital": dict(min_value=0.0, max_value=1.0, step=0.01),
        "Owner share capital": dict(min_value=0.0, max_value=1.0, step=0.01),
        "Discount rate": dict(min_value=0.0, max_value=0.5, step=0.01),
    }

    for parameter, cfg in slider_cfg.items():
        if parameter in df["Parameter"].values:
            idx = df.index[df["Parameter"] == parameter][0]
            state_key = f"key_assumption_{parameter.replace(' ', '_').lower()}"
            value_key = f"{state_key}_value"
            min_value = float(cfg["min_value"])
            max_value = float(cfg["max_value"])
            step = float(cfg["step"])

            if value_key not in st.session_state:
                st.session_state[value_key] = float(df.at[idx, "Value"])

            minus_col, value_col, plus_col = st.columns([1, 4, 1])
            with minus_col:
                if st.button("➖", key=f"{state_key}_minus"):
                    st.session_state[value_key] = max(
                        min_value, st.session_state[value_key] - step
                    )
            with plus_col:
                if st.button("➕", key=f"{state_key}_plus"):
                    st.session_state[value_key] = min(
                        max_value, st.session_state[value_key] + step
                    )

            current_value = value_col.number_input(
                parameter,
                min_value=min_value,
                max_value=max_value,
                step=float(step),
                value=float(st.session_state[value_key]),
                key=f"{state_key}_input",
            )
            st.session_state[value_key] = float(current_value)
            df.at[idx, "Value"] = float(current_value)
    table.data = df


def _numeric_step(value: float) -> float:
    """Return a sensible increment for Streamlit number inputs."""

    if value is None or pd.isna(value):
        return 0.01
    value = abs(float(value))
    if value == 0:
        return 0.01
    exponent = max(-2, int(np.floor(np.log10(value))) - 1)
    return round(10 ** exponent, 6)


def _modify_default_inputs(page: InputLandingPage) -> None:
    """Allow users to tweak any default input/figure via focused controls."""

    st.subheader("Modify Default Inputs & Figures")
    tables = page.tables()
    if not tables:
        st.info("No tables are available to edit.")
        return

    table_names = list(tables.keys())
    table_name = st.selectbox(
        "Select table",
        table_names,
        key="default_table_select",
    )
    table = tables[table_name]

    if table.data.empty:
        st.info("The selected table has no rows to modify. Use the table editor below to add data.")
        return

    id_column = table.columns[0] if table.columns else None
    row_indices = list(table.data.index)

    def _format_row(idx: int) -> str:
        if id_column and id_column in table.data.columns:
            label = table.data.at[idx, id_column]
            if label is None or pd.isna(label) or str(label).strip() == "":
                label = f"Row {idx + 1}"
        else:
            label = f"Row {idx + 1}"
        return f"{idx + 1}. {label}"

    row_idx = st.selectbox(
        "Select row",
        row_indices,
        format_func=_format_row,
        key=f"default_row_select_{table_name}",
    )

    st.markdown("Adjust the values for the selected row:")
    for column in table.columns:
        current_value = table.data.at[row_idx, column]
        widget_key = f"default_edit_{table_name}_{row_idx}_{column}".replace(" ", "_").lower()

        numeric_series = pd.to_numeric(table.data[column], errors="coerce")
        is_numeric = pd.api.types.is_numeric_dtype(table.data[column]) or numeric_series.notna().any()

        if is_numeric:
            if row_idx in numeric_series.index:
                base_value = numeric_series.loc[row_idx]
            else:
                base_value = numeric_series.iloc[0] if not numeric_series.empty else 0.0
            if pd.isna(base_value):
                base_value = 0.0
            # Streamlit number_input requires all numeric arguments to share the
            # same underlying type (ints vs floats). ``_numeric_step`` can
            # return an integer when the magnitude is large, so coerce the
            # step to ``float`` to avoid "mixed numeric types" errors when the
            # value is a float.
            step = float(_numeric_step(base_value))
            number_format = "%.0f" if step >= 1 else "%.4f"
            new_value = st.number_input(
                column,
                value=float(base_value),
                step=step,
                format=number_format,
                key=widget_key,
            )
            if pd.api.types.is_integer_dtype(table.data[column]):
                new_value = int(round(new_value))
            table.data.at[row_idx, column] = new_value
        else:
            text_value = "" if current_value is None or pd.isna(current_value) else str(current_value)
            new_value = st.text_input(
                column,
                value=text_value,
                key=widget_key,
            )
            table.data.at[row_idx, column] = new_value

    st.caption("Updates are applied immediately. Use the section tables below for bulk edits or row management.")


def _editable_tables(page: InputLandingPage) -> None:
    """Render editable data tables grouped by the landing-page sections."""

    categories = page.grouped_tables()
    tabs = st.tabs(list(categories.keys()))

    for tab, (section, tables) in zip(tabs, categories.items()):
        with tab:
            for table in tables:
                expanded = section in {"Global", "Capex", "Financial"}
                _render_table(table, expanded=expanded)
                if table is page.initial_investment:
                    st.metric("Total Initial Investment", _format_currency(page.total_initial_investment))


def _render_table(table: EditableTable, expanded: bool = False) -> None:
    """Show a Streamlit data editor for a specific table."""

    safe_key = f"table_{table.name.replace(' ', '_').lower()}"
    with st.expander(table.name, expanded=expanded):
        controls = st.columns(2)
        if controls[0].button("➕ Add row", key=f"add_{safe_key}"):
            table.add_row({column: None for column in table.columns})
            st.experimental_rerun()

        if not table.data.empty:
            row_options = list(table.data.index)
            remove_index = controls[1].selectbox(
                "Row to remove",
                options=row_options,
                key=f"remove_select_{safe_key}",
                label_visibility="collapsed",
                format_func=lambda i: f"Row {i + 1}",
            )
            if controls[1].button("➖ Remove selected", key=f"remove_{safe_key}"):
                table.remove_row(int(remove_index))
                st.experimental_rerun()
        else:
            controls[1].markdown("&nbsp;")

        edited = st.data_editor(
            table.data,
            num_rows="dynamic",
            use_container_width=True,
            key=safe_key,
        )
        if isinstance(edited, pd.DataFrame):
            table.data = edited[table.columns].copy()
        else:  # pragma: no cover - safety for older Streamlit returning list of dicts
            table.data = pd.DataFrame(edited, columns=table.columns)

def _annualise(rate: float | None) -> float | None:
    if rate is None or pd.isna(rate):
        return None
    return (1 + rate) ** 12 - 1


def _format_rate(rate: float | None) -> str:
    if rate is None or pd.isna(rate):
        return "n/a"
    return f"{rate * 100:,.2f}%"


def _format_percent(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value * 100:,.1f}%"


def _format_currency(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"${value:,.0f}"


def _reset_period_index(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=[label])
    result = df.copy().reset_index()
    if result.columns[0] != label:
        result = result.rename(columns={result.columns[0]: label})
    if np.issubdtype(result[label].dtype, np.datetime64):
        result[label] = pd.to_datetime(result[label]).dt.to_period("M").astype(str)
    return result

def _render_key_metrics(model: CassavaBioethanolModel, results: Dict[str, object]) -> None:
    metrics = results["metrics"]
    revenue = results["revenue"]
    production = results["production"]
    costs = results["costs"]
    financials = results["financials"]
    loan_schedule = results["loan_schedule"]
    depreciation = results["depreciation"]

    st.subheader("Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Project NPV", _format_currency(metrics.get("Project NPV")))
    col2.metric("Project IRR (annual)", _format_rate(_annualise(metrics.get("Project IRR"))))
    col3.metric("Equity IRR (annual)", _format_rate(_annualise(metrics.get("Equity IRR"))))
    payback_years = metrics.get("Payback Period (years)")
    col4.metric("Payback Period (years)", f"{payback_years:,.1f}" if payback_years and not pd.isna(payback_years) else "n/a")

    st.markdown("### Assumptions Snapshot")
    assumption_snapshot = pd.DataFrame(
        {
            "Assumption": [
                "Corporate Tax Rate",
                "Investor Share",
                "Owner Share",
                "Discount Rate",
                "Terminal Growth Rate",
                "Capital Gains Tax Rate",
                "Total Initial Investment",
            ],
            "Value": [
                _format_rate(metrics.get("Corporate Tax Rate")),
                _format_percent(metrics.get("Investor Share")),
                _format_percent(metrics.get("Owner Share")),
                _format_rate(metrics.get("Discount Rate")),
                _format_rate(metrics.get("Terminal Growth Rate")),
                _format_rate(metrics.get("Capital Gains Tax Rate")),
                _format_currency(metrics.get("Total Initial Investment")),
            ],
        }
    )
    st.dataframe(assumption_snapshot, use_container_width=True, hide_index=True)

    st.markdown("### Latest Drivers")
    drivers = pd.DataFrame(
        {
            "Metric": [
                "Final Month Revenue",
                "Final Month EBITDA",
                "Final Month Equity Cash Flow",
                "Cumulative FCF",
                "Cumulative Equity CF",
            ],
            "Value": [
                _format_currency(metrics.get("Final Month Revenue")),
                _format_currency(metrics.get("Final Month EBITDA")),
                _format_currency(metrics.get("Final Month Equity CF")),
                _format_currency(metrics.get("Cumulative FCF")),
                _format_currency(metrics.get("Cumulative Equity CF")),
            ],
        }
    )
    st.dataframe(drivers, use_container_width=True, hide_index=True)

    st.markdown("### Annual Operations & Production")
    production_annual = production.annual.copy()
    if not production_annual.empty:
        st.bar_chart(production_annual)
    else:
        st.info("No production data available for the selected horizon.")

    summary_cols = [col for col in ["Revenue", "EBITDA", "Net Income"] if col in financials.income_annual.columns]
    if summary_cols:
        annual_summary = financials.income_annual[summary_cols].copy()
        annual_summary.index.name = "Year"
        st.dataframe(annual_summary.reset_index(), use_container_width=True)

    st.markdown("### Cash Flow & Returns")
    cash_columns = [c for c in ["Operating Cash Flow", "Free Cash Flow", "Equity Cash Flow"] if c in financials.cashflow_monthly.columns]
    if cash_columns:
        st.line_chart(financials.cashflow_monthly[cash_columns])
        cumulative_df = financials.cashflow_monthly[cash_columns].cumsum()
        cumulative_df.columns = [f"Cumulative {col}" for col in cumulative_df.columns]
        st.line_chart(cumulative_df)

    st.markdown("### Revenue Mix")
    revenue_annual = revenue.annual.copy()
    if not revenue_annual.empty:
        if "Total Revenue" in revenue_annual:
            mix_df = revenue_annual.drop(columns=["Total Revenue"])
        else:
            mix_df = revenue_annual
        if not mix_df.empty:
            st.bar_chart(mix_df)
        st.dataframe(revenue_annual.reset_index().rename(columns={"index": "Year"}), use_container_width=True)
    else:
        st.info("Revenue inputs are empty for the current projection.")

    st.markdown("### Operating Cost Breakdown")
    cost_monthly = pd.DataFrame(
        {
            name: output.monthly.sum(axis=1)
            for name, output in costs.items()
            if output and not output.monthly.empty
        }
    )
    if not cost_monthly.empty:
        st.area_chart(cost_monthly)
    cost_annual = pd.DataFrame(
        {
            name: output.annual.sum(axis=1)
            for name, output in costs.items()
            if output and not output.annual.empty
        }
    )
    if not cost_annual.empty:
        cost_annual.index.name = "Year"
        st.dataframe(cost_annual.reset_index(), use_container_width=True)

    st.markdown("### Capital Expenditure & Debt")
    capex_df = model.input_page.initial_investment.data.copy()
    if not capex_df.empty:
        st.bar_chart(capex_df.set_index("Item")["Cost"])
        st.dataframe(capex_df, use_container_width=True)
    debt_chart = loan_schedule.schedule.pivot_table(index="Month", values="Closing Balance", aggfunc="sum")
    if not debt_chart.empty:
        st.line_chart(debt_chart)

    st.markdown("### Fixed Asset Summary")
    st.dataframe(depreciation.summary, use_container_width=True)

    st.markdown("### Debt Schedule Chart")
    debt_payments = loan_schedule.schedule.pivot_table(index="Month", values="Payment", aggfunc="sum")
    if not debt_payments.empty:
        st.area_chart(debt_payments)

    st.markdown("### Break-even Analysis")
    break_even_df = results.get("break_even")
    if isinstance(break_even_df, pd.DataFrame) and not break_even_df.empty:
        st.dataframe(_reset_period_index(break_even_df, "Month"), use_container_width=True)

def _render_financial_performance(results: Dict[str, object]) -> None:
    financials = results["financials"]
    costs = results["costs"]

    st.subheader("Monthly Financial Performance")
    st.dataframe(_reset_period_index(financials.income_monthly, "Month"), use_container_width=True)

    st.subheader("Annual Financial Performance")
    annual_income = financials.income_annual.copy()
    annual_income.index.name = "Year"
    st.dataframe(annual_income.reset_index(), use_container_width=True)

    st.subheader("Total Expense Schedule")
    monthly_expense = pd.DataFrame(
        {
            name: output.monthly.sum(axis=1)
            for name, output in costs.items()
            if output and not output.monthly.empty
        }
    )
    if not monthly_expense.empty:
        st.dataframe(_reset_period_index(monthly_expense, "Month"), use_container_width=True)
    annual_expense = pd.DataFrame(
        {
            name: output.annual.sum(axis=1)
            for name, output in costs.items()
            if output and not output.annual.empty
        }
    )
    if not annual_expense.empty:
        annual_expense.index.name = "Year"
        st.dataframe(annual_expense.reset_index(), use_container_width=True)

def _render_financial_position(results: Dict[str, object]) -> None:
    financials = results["financials"]

    st.subheader("Monthly Statement of Financial Position")
    st.dataframe(_reset_period_index(financials.balance_monthly, "Month"), use_container_width=True)

    st.subheader("Annual Statement of Financial Position")
    balance_annual = financials.balance_annual.copy()
    balance_annual.index.name = "Year"
    st.dataframe(balance_annual.reset_index(), use_container_width=True)

def _render_cash_flow_page(results: Dict[str, object]) -> None:
    financials = results["financials"]

    st.subheader("Monthly Cash Flow Statement")
    cash_monthly = financials.cashflow_monthly
    st.dataframe(_reset_period_index(cash_monthly, "Month"), use_container_width=True)

    st.subheader("Annual Cash Flow Statement")
    cash_annual = financials.cashflow_annual.copy()
    cash_annual.index.name = "Year"
    st.dataframe(cash_annual.reset_index(), use_container_width=True)

    st.subheader("Cash Flow and Returns Charts")
    cash_columns = [c for c in ["Operating Cash Flow", "Free Cash Flow", "Equity Cash Flow"] if c in cash_monthly.columns]
    if cash_columns:
        st.line_chart(cash_monthly[cash_columns])

    st.subheader("Cumulative Equity Cash Flow")
    if "Equity Cash Flow" in cash_monthly.columns:
        cumulative_equity = cash_monthly[["Equity Cash Flow"]].cumsum()
        cumulative_equity.columns = ["Cumulative Equity Cash Flow"]
        st.line_chart(cumulative_equity)
        cumulative_df = cumulative_equity.reset_index().rename(columns={cumulative_equity.index.name or "index": "Month"})
        st.dataframe(cumulative_df, use_container_width=True)

def _render_sensitivity_page(model: CassavaBioethanolModel) -> None:
    st.subheader("Sensitivity Analysis Configuration")
    config_df = pd.DataFrame([s.__dict__ for s in DEFAULT_SENSITIVITY_SCENARIOS]) if DEFAULT_SENSITIVITY_SCENARIOS else pd.DataFrame(columns=["name", "parameter", "delta"])
    st.dataframe(config_df.rename(columns={"name": "Scenario", "parameter": "Parameter", "delta": "Delta"}), use_container_width=True, hide_index=True)

    analysis_model = CassavaBioethanolModel(copy.deepcopy(model.input_page))
    if DEFAULT_SENSITIVITY_SCENARIOS:
        sensitivity_results = run_sensitivity(analysis_model, DEFAULT_SENSITIVITY_SCENARIOS)
    else:
        sensitivity_results = pd.DataFrame(columns=["Scenario", "Parameter", "Delta", "Project NPV", "Change vs Base"])
    st.subheader("Simulation Results")
    st.dataframe(sensitivity_results, use_container_width=True)

    st.subheader("Monte Carlo Simulation Configuration")
    mc_rows = (
        [{"Setting": "Iterations", "Value": MONTE_CARLO_ITERATIONS}, {"Setting": "Random Seed", "Value": MONTE_CARLO_SEED}]
        + [{"Setting": f"Std Dev - {param}", "Value": std} for param, std in MONTE_CARLO_STD.items()]
    )
    st.dataframe(pd.DataFrame(mc_rows), use_container_width=True, hide_index=True)

    st.subheader("Tornado Drivers")
    tornado_model = CassavaBioethanolModel(copy.deepcopy(model.input_page))
    tornado_df = tornado_chart_inputs(tornado_model, TORNADO_DRIVERS, scale=0.1)
    st.dataframe(tornado_df, use_container_width=True)

def _render_scenario_page(model: CassavaBioethanolModel, results: Dict[str, object]) -> None:
    st.subheader("Scenario/Is Configuration")
    scenario_df = pd.DataFrame([{"Scenario": cfg.name, **cfg.overrides} for cfg in DEFAULT_SCENARIO_CONFIGS]) if DEFAULT_SCENARIO_CONFIGS else pd.DataFrame(columns=["Scenario"])
    st.dataframe(scenario_df, use_container_width=True)

    st.subheader("Scenario Tool Configuration")
    tool_df = model.input_page.global_inputs.data.rename(columns={"Value": "Base Value"}).copy()
    numeric_values = pd.to_numeric(tool_df["Base Value"], errors="coerce")
    tool_df["Low Bound"] = np.where(numeric_values.notna(), numeric_values * 0.8, np.nan)
    tool_df["High Bound"] = np.where(numeric_values.notna(), numeric_values * 1.2, np.nan)
    desired_order = ["Parameter", "Base Value", "Units", "Low Bound", "High Bound"]
    tool_df = tool_df[[c for c in desired_order if c in tool_df.columns]]
    st.dataframe(tool_df, use_container_width=True, hide_index=True)

    comparison_model = CassavaBioethanolModel(copy.deepcopy(model.input_page))
    if DEFAULT_SCENARIO_CONFIGS:
        comparison_df = scenario_comparison(comparison_model, DEFAULT_SCENARIO_CONFIGS)
    else:
        comparison_df = pd.DataFrame(columns=["Scenario", "Project NPV", "Project IRR", "Equity IRR"])
    st.subheader("Scenario Comparison")
    st.dataframe(comparison_df, use_container_width=True)

    st.subheader("Goal Seek Results")
    goal_seek_parameter = "Corporate tax rate"
    goal_seek_metric = "Project NPV"
    try:
        target_value = float(results["metrics"].get(goal_seek_metric, 0.0))
        goal_model = CassavaBioethanolModel(copy.deepcopy(model.input_page))
        goal_result = goal_seek_to_target(goal_model, goal_seek_parameter, goal_seek_metric, target_value)
        goal_df = pd.DataFrame(
            [
                {
                    "Parameter": goal_seek_parameter,
                    "Target Metric": goal_seek_metric,
                    "Target Value": target_value,
                    "Target Name": goal_result.target_name,
                    "Achieved Value": goal_result.achieved_value,
                    "Tolerance": goal_result.tolerance,
                    "Iterations": goal_result.iterations,
                }
            ]
        )
    except KeyError:
        goal_df = pd.DataFrame(columns=["Parameter", "Target Metric", "Target Value", "Target Name", "Achieved Value", "Tolerance", "Iterations"])
    st.dataframe(goal_df, use_container_width=True)

def _render_monte_carlo_page(model: CassavaBioethanolModel) -> None:
    current_version = st.session_state.get("model_version")
    cache_version = st.session_state.get("mc_cache_version")
    if st.session_state.get("mc_cache") is None or cache_version != current_version:
        mc_model = CassavaBioethanolModel(copy.deepcopy(model.input_page))
        st.session_state.mc_cache = monte_carlo_simulation(
            mc_model,
            parameter_std=MONTE_CARLO_STD,
            iterations=MONTE_CARLO_ITERATIONS,
            random_seed=MONTE_CARLO_SEED,
        )
        st.session_state.mc_cache_version = current_version
    mc_results = st.session_state.get("mc_cache", pd.DataFrame())

    st.subheader("Monte Carlo Simulation Results")
    if mc_results.empty:
        st.info("Monte Carlo results are not available. Adjust the configuration or recalculate the model.")
        return

    summary = mc_results.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).T
    st.dataframe(summary, use_container_width=True)

    st.subheader("NPV Distribution (sorted path)")
    st.line_chart(mc_results["Project NPV"].sort_values().reset_index(drop=True))

    st.subheader("IRR Distribution (sorted path)")
    if "Project IRR" in mc_results:
        st.line_chart(mc_results["Project IRR"].sort_values().reset_index(drop=True))

def main() -> None:
    st.title("Cassava_Bioethanol Financial Model")
    st.caption("Adjust the assumptions, run the project finance model, and inspect the outputs across dedicated dashboards.")

    input_page = _load_session_inputs()

    action_cols = st.columns([1, 1])
    with action_cols[0]:
        recalc = st.button("Recalculate model", type="primary")

    if recalc or "model_results" not in st.session_state:
        model, results = _build_model_snapshot(input_page)
        st.session_state.model_results = (model, results)
        st.session_state.model_version = st.session_state.get("model_version", 0) + 1
        st.session_state.mc_cache = None
        st.session_state.mc_cache_version = None
        st.session_state.excel_bytes = _generate_excel_bytes(model)

    model, results = st.session_state.model_results

    with action_cols[1]:
        excel_bytes = st.session_state.get("excel_bytes")
        if excel_bytes:
            st.download_button(
                "Download Excel Model",
                data=excel_bytes,
                file_name="Cassava_Bioethanol_Financial_Model.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    st.caption("Use the navigation tabs to move between the input landing page and the analytical dashboards.")

    tabs = st.tabs(
        [
            "Input Landing Page",
            "Key Metrics Dashboard",
            "Financial Performance",
            "Financial Position",
            "Cash Flow Statement",
            "Sensitivity Analyses",
            "Scenario / IFs Analysis",
            "Monte Carlo Simulation",
        ]
    )

    with tabs[0]:
        st.subheader("Input Landing Page")
        st.info("Edit the assumptions and press 'Recalculate model' to refresh the financial outputs.")
        _update_projection(input_page)
        _key_assumptions_controls(input_page.global_inputs)
        _modify_default_inputs(input_page)
        _editable_tables(input_page)

    with tabs[1]:
        _render_key_metrics(model, results)

    with tabs[2]:
        _render_financial_performance(results)

    with tabs[3]:
        _render_financial_position(results)

    with tabs[4]:
        _render_cash_flow_page(results)

    with tabs[5]:
        _render_sensitivity_page(model)

    with tabs[6]:
        _render_scenario_page(model, results)

    with tabs[7]:
        _render_monte_carlo_page(model)


if __name__ == "__main__":
    main()
