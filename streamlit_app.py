"""Streamlit interface for the Cassava bioethanol financial model."""
from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Dict

import pandas as pd
import streamlit as st

from bioethanol_model import CassavaBioethanolModel
from bioethanol_model.exporter import export_to_excel
from bioethanol_model.inputs import EditableTable, InputLandingPage, default_input_page


st.set_page_config(page_title="Cassava Bioethanol Model", layout="wide")


def _load_session_inputs() -> InputLandingPage:
    """Return the mutable input landing page stored in session state."""
    if "input_page" not in st.session_state:
        st.session_state.input_page = default_input_page()
    return st.session_state.input_page


def _update_projection(page: InputLandingPage) -> None:
    """Render projection horizon controls in the sidebar and update in place."""
    with st.sidebar:
        st.header("Projection Horizon")
        start = st.number_input(
            "Start Year",
            min_value=2000,
            max_value=2100,
            value=int(page.projection.start_year),
            step=1,
        )
        end = st.number_input(
            "End Year",
            min_value=start,
            max_value=2125,
            value=int(page.projection.end_year),
            step=1,
        )
        page.projection.start_year = int(start)
        page.projection.end_year = int(end)


def _sidebar_key_assumptions(table: EditableTable) -> None:
    """Expose frequently tweaked assumptions with Streamlit widgets."""
    with st.sidebar:
        st.header("Key Assumptions")
        df = table.data.copy()
        for parameter, cfg in {
            "Corporate tax rate": dict(min_value=0.0, max_value=0.7, step=0.01),
            "Investor share capital": dict(min_value=0.0, max_value=1.0, step=0.01),
            "Owner share capital": dict(min_value=0.0, max_value=1.0, step=0.01),
            "Discount rate": dict(min_value=0.0, max_value=0.5, step=0.01),
        }.items():
            if parameter in df["Parameter"].values:
                idx = df.index[df["Parameter"] == parameter][0]
                current = float(df.at[idx, "Value"])
                df.at[idx, "Value"] = st.slider(parameter, value=current, **cfg)
        table.data = df


def _editable_tables(page: InputLandingPage) -> None:
    """Render editable data tables grouped by the landing-page sections."""

    categories = page.grouped_tables()
    tabs = st.tabs(list(categories.keys()))

    for tab, (section, tables) in zip(tabs, categories.items()):
        with tab:
            if section == "Global":
                st.subheader("Projection Horizon")
                st.table(page.projection.to_frame())
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


def _format_currency(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"${value:,.0f}"


@st.cache_data(show_spinner=False)
def _excel_download_bytes(page: InputLandingPage) -> bytes:
    model = CassavaBioethanolModel(page)
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir) / "Cassava_Bioethanol_Financial_Model.xlsx"
        export_to_excel(model, temp_path)
        return temp_path.read_bytes()


def _display_outputs(model: CassavaBioethanolModel, results: Dict[str, object]) -> None:
    metrics = results["metrics"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Project NPV", _format_currency(metrics.get("Project NPV")))
    col2.metric("Project IRR (annual)", _format_rate(_annualise(metrics.get("Project IRR"))))
    col3.metric("Equity IRR (annual)", _format_rate(_annualise(metrics.get("Equity IRR"))))

    col4, col5, col6 = st.columns(3)
    col4.metric("Cumulative FCF", _format_currency(metrics.get("Cumulative FCF")))
    col5.metric("Cumulative Equity CF", _format_currency(metrics.get("Cumulative Equity CF")))
    col6.metric("Final Month EBITDA", _format_currency(metrics.get("Final Month EBITDA")))

    revenue = results["revenue"].monthly.copy()
    costs = pd.concat({name: output.monthly.sum(axis=1) for name, output in results["costs"].items()}, axis=1)
    financials = results["financials"]
    working_capital = results["working_capital"].monthly
    loan_schedule = results["loan_schedule"].schedule

    tabs = st.tabs(["Revenue", "Operating Costs", "Financial Statements", "Working Capital", "Debt", "Break-even"])

    with tabs[0]:
        st.subheader("Monthly Revenue")
        st.line_chart(revenue["Total Revenue"])
        st.dataframe(revenue.reset_index().rename(columns={"index": "Month"}), use_container_width=True)

    with tabs[1]:
        st.subheader("Operating Cost Breakdown")
        st.area_chart(costs)
        st.dataframe(costs.reset_index().rename(columns={"index": "Month"}), use_container_width=True)

    with tabs[2]:
        st.subheader("Income Statement (Monthly)")
        st.dataframe(
            financials.income_monthly.reset_index().rename(columns={"index": "Month"}),
            use_container_width=True,
        )
        st.subheader("Cash Flow Statement (Monthly)")
        st.dataframe(
            financials.cashflow_monthly.reset_index().rename(columns={"index": "Month"}),
            use_container_width=True,
        )
        st.subheader("Balance Sheet (Monthly)")
        st.dataframe(
            financials.balance_monthly.reset_index().rename(columns={"index": "Month"}),
            use_container_width=True,
        )

    with tabs[3]:
        st.subheader("Working Capital")
        st.line_chart(working_capital[["Receivables", "Inventory", "Payables"]])
        st.dataframe(working_capital.reset_index().rename(columns={"index": "Month"}), use_container_width=True)

    with tabs[4]:
        st.subheader("Loan Schedule")
        st.dataframe(loan_schedule.reset_index(drop=True), use_container_width=True)

    with tabs[5]:
        st.subheader("Break-even Analysis")
        break_even = results["break_even"].copy()
        st.line_chart(break_even[["Monthly Margin", "Cumulative Margin"]])
        st.dataframe(break_even.reset_index().rename(columns={"index": "Month"}), use_container_width=True)

    excel_bytes = _excel_download_bytes(model.input_page)
    st.download_button(
        "Download Excel Model",
        data=excel_bytes,
        file_name="Cassava_Bioethanol_Financial_Model.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def main() -> None:
    st.title("Cassava Bioethanol Financial Model Dashboard")
    st.write(
        "Adjust the modelling assumptions below and recalculate the integrated financial statements, key metrics, and break-even"
        " diagnostics for the cassava-based ethanol project."
    )

    input_page = _load_session_inputs()
    _update_projection(input_page)
    _sidebar_key_assumptions(input_page.global_inputs)

    st.header("Input Landing Page")
    _editable_tables(input_page)

    if st.button("Recalculate model", type="primary") or "model_results" not in st.session_state:
        model = CassavaBioethanolModel(input_page)
        st.session_state.model_results = (model, model.build())

    if "model_results" in st.session_state:
        model, results = st.session_state.model_results
        st.header("Outputs")
        _display_outputs(model, results)


if __name__ == "__main__":
    main()
