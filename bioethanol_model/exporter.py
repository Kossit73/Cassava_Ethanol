from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from .financial_model import CassavaBioethanolModel
from .inputs import InputLandingPage
from .scenario import ScenarioConfig, goal_seek_to_target, scenario_comparison
from .sensitivity import SensitivityScenario, monte_carlo_simulation, run_sensitivity, tornado_chart_inputs


SECTION_GAP = 2


def _write_table(
    writer: pd.ExcelWriter,
    sheet: str,
    df: pd.DataFrame,
    title: str,
    startrow: int = 0,
    startcol: int = 0,
    *,
    index: bool = True,
) -> int:
    if sheet not in writer.sheets:
        worksheet = writer.book.add_worksheet(sheet)
        writer.sheets[sheet] = worksheet
    else:
        worksheet = writer.sheets[sheet]
    worksheet.write_string(startrow, startcol, title)
    df_to_write = df.copy()
    df_to_write.to_excel(
        writer,
        sheet_name=sheet,
        startrow=startrow + 1,
        startcol=startcol,
        index=index,
    )
    return startrow + len(df_to_write.index) + SECTION_GAP + 2


def export_to_excel(
    model: CassavaBioethanolModel,
    output_path: Path,
    sensitivity_scenarios: Iterable[SensitivityScenario] | None = None,
    scenario_configs: Iterable[ScenarioConfig] | None = None,
) -> Path:
    output_path = Path(output_path)
    results = model.build()

    sensitivity_scenarios = list(sensitivity_scenarios or [])
    scenario_configs = list(scenario_configs or [])

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        _write_input_page(writer, model.input_page)
        _write_key_metrics(writer, model, results)
        _write_financial_performance(writer, results)
        _write_financial_position(writer, results)
        _write_cash_flow_page(writer, results)
        _write_sensitivity_page(writer, model, sensitivity_scenarios)
        _write_scenario_page(writer, model, scenario_configs)
        _write_break_even_page(writer, results)
    return output_path


def _write_input_page(writer: pd.ExcelWriter, page: InputLandingPage) -> None:
    sheet = "Input Landing"
    projection_df = page.projection.to_frame()
    next_row = _write_table(writer, sheet, projection_df, "Projection Horizon")
    for title, table in page.tables().items():
        next_row = _write_table(writer, sheet, table.data, title, startrow=next_row)


def _write_key_metrics(writer: pd.ExcelWriter, model: CassavaBioethanolModel, results: Dict[str, object]) -> None:
    sheet = "Key Metrics"
    if sheet in writer.sheets:
        worksheet = writer.sheets[sheet]
    else:
        worksheet = writer.book.add_worksheet(sheet)
        writer.sheets[sheet] = worksheet

    metrics = results["metrics"]
    projection = model.input_page.projection

    def _get_metric(name: str, default=np.nan) -> float:
        value = metrics.get(name, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return value

    def _annualise(rate: float) -> float:
        try:
            if rate is None or not np.isfinite(rate):
                return np.nan
            return (1 + rate) ** 12 - 1
        except Exception:
            return np.nan

    assumptions_snapshot = pd.DataFrame(
        {
            "Value": [
                projection.start_year,
                projection.end_year,
                projection.end_year - projection.start_year + 1,
                _get_metric("Discount Rate"),
                _get_metric("Terminal Growth Rate"),
                _get_metric("Capital Gains Tax Rate"),
                _get_metric("Total Initial Investment"),
            ]
        },
        index=[
            "Start Year",
            "End Year",
            "Projection Years",
            "Discount Rate",
            "Terminal Growth Rate",
            "Capital Gains Tax Rate",
            "Total Initial Investment",
        ],
    )

    global_section = pd.DataFrame(
        {
            "Value": [
                _get_metric("Corporate Tax Rate"),
                _get_metric("Investor Share"),
                _get_metric("Owner Share"),
                _get_metric("Payback Period (years)"),
                metrics.get("Payback Month", "N/A"),
            ]
        },
        index=[
            "Corporate Tax Rate",
            "Investor Share",
            "Owner Share",
            "Payback Period (years)",
            "Payback Month",
        ],
    )

    latest = pd.DataFrame(
        {
            "Latest": {
                "Final Month Revenue": _get_metric("Final Month Revenue"),
                "Final Month EBITDA": _get_metric("Final Month EBITDA"),
                "Final Month Equity CF": _get_metric("Final Month Equity CF"),
                "Cumulative FCF": _get_metric("Cumulative FCF"),
                "Cumulative Equity CF": _get_metric("Cumulative Equity CF"),
            }
        }
    )

    overview = pd.DataFrame(
        {
            "Value": {
                "Project NPV": _get_metric("Project NPV"),
                "Project IRR (annual)": _annualise(_get_metric("Project IRR")),
                "Equity IRR (annual)": _annualise(_get_metric("Equity IRR")),
                "Investor IRR (annual)": _annualise(_get_metric("Investor IRR")),
                "Owner IRR (annual)": _annualise(_get_metric("Owner IRR")),
                "Payback Period (years)": _get_metric("Payback Period (years)"),
            }
        }
    )

    top_left_end = _write_table(writer, sheet, assumptions_snapshot, "Assumptions Snapshot", startrow=0, startcol=0)
    top_mid_end = _write_table(writer, sheet, global_section, "Global Summary", startrow=0, startcol=4)
    top_right_end = _write_table(writer, sheet, latest, "Latest Drivers", startrow=0, startcol=8)
    top_end = max(top_left_end, top_mid_end, top_right_end)

    overview_end = _write_table(writer, sheet, overview, "Overview", startrow=top_end, startcol=0)

    income_annual = results["financials"].income_annual[["Revenue", "EBITDA", "Net Income"]]
    annual_ops = pd.concat(
        [
            income_annual,
            results["production"].annual,
        ],
        axis=1,
    ).reset_index().rename(columns={"index": "Year"})
    annual_ops_end = _write_table(
        writer,
        sheet,
        annual_ops,
        "Annual Operations & Production Summary",
        startrow=overview_end,
        startcol=0,
        index=False,
    )

    fixed_asset = results["depreciation"].summary.set_index("Item")
    fixed_end = _write_table(
        writer,
        sheet,
        fixed_asset,
        "Fixed Asset Summary",
        startrow=top_end,
        startcol=4,
    )

    current_row = max(fixed_end, annual_ops_end)
    chart_col = 8
    chart_height = 18

    def _write_chart_table(
        df: pd.DataFrame,
        title: str,
        chart_type: str,
        *,
        categories_col: int = 0,
        exclude_columns: Iterable[str] | None = None,
        subtype: str | None = None,
        insert_kwargs: Dict[str, float] | None = None,
    ) -> None:
        nonlocal current_row
        data = df.copy()
        startcol = 0
        table_end = _write_table(
            writer,
            sheet,
            data,
            title,
            startrow=current_row,
            startcol=startcol,
            index=False,
        )
        header_row = current_row + 1
        data_start = current_row + 2
        data_end = current_row + 1 + len(data.index)
        chart = writer.book.add_chart({"type": chart_type} if subtype is None else {"type": chart_type, "subtype": subtype})
        cols = list(range(data.shape[1]))
        if exclude_columns:
            cols = [c for c in cols if data.columns[c] not in exclude_columns]
        if len(cols) <= 1:
            current_row = max(table_end, current_row + chart_height)
            return
        for col_idx in cols:
            if col_idx == categories_col:
                continue
            chart.add_series(
                {
                    "name": [sheet, header_row, startcol + col_idx],
                    "categories": [sheet, data_start, startcol + categories_col, data_end, startcol + categories_col],
                    "values": [sheet, data_start, startcol + col_idx, data_end, startcol + col_idx],
                }
            )
        chart.set_title({"name": title})
        chart.set_x_axis({"name": data.columns[categories_col]})
        chart.set_y_axis({"major_gridlines": {"visible": True}})
        chart.set_legend({"position": "bottom"})
        worksheet.insert_chart(
            current_row,
            chart_col,
            chart,
            insert_kwargs or {"x_scale": 1.1, "y_scale": 1.1},
        )
        current_row = max(table_end, current_row + chart_height)

    cash_monthly = results["financials"].cashflow_monthly
    if not cash_monthly.empty:
        month_labels = cash_monthly.index.to_period("M").astype(str)
        cf_columns = [
            col
            for col in ["Operating Cash Flow", "Free Cash Flow", "Equity Cash Flow"]
            if col in cash_monthly.columns
        ]
        if cf_columns:
            cash_returns_df = cash_monthly[cf_columns].copy()
            cash_returns_df.insert(0, "Month", month_labels)
            _write_chart_table(cash_returns_df, "Cash Flow & Returns", "column")

        cumulative_series: Dict[str, pd.Series] = {}
        if "Free Cash Flow" in cash_monthly.columns:
            cumulative_series["Cumulative Free Cash Flow"] = cash_monthly["Free Cash Flow"].cumsum()
        if "Equity Cash Flow" in cash_monthly.columns:
            cumulative_series["Cumulative Equity Cash Flow"] = cash_monthly["Equity Cash Flow"].cumsum()
        if cumulative_series:
            cumulative_chart_df = pd.DataFrame({"Month": month_labels})
            for name, series in cumulative_series.items():
                cumulative_chart_df[name] = series.values
            _write_chart_table(cumulative_chart_df, "Cumulative Cash Flows", "line")

    production_df = results["production"].annual.reset_index().rename(columns={"index": "Year"})
    if not production_df.empty:
        _write_chart_table(production_df, "Annual Production", "line")

    cashflow_annual = results["financials"].cashflow_annual.reset_index().rename(columns={"index": "Year"})
    cash_columns = ["Operating Cash Flow", "Free Cash Flow", "Equity Cash Flow"]
    cash_columns = [c for c in cash_columns if c in cashflow_annual.columns]
    if cash_columns:
        cash_df = cashflow_annual[["Year", *cash_columns]]
        _write_chart_table(cash_df, "Cash Flow Summary", "column")

    revenue_df = results["revenue"].annual.reset_index().rename(columns={"index": "Year"})
    if not revenue_df.empty:
        exclude = ["Total Revenue"] if "Total Revenue" in revenue_df.columns else None
        _write_chart_table(
            revenue_df,
            "Revenue Mix",
            "column",
            subtype="stacked",
            exclude_columns=exclude,
        )

    cost_totals = {
        name: output.annual.sum(axis=1)
        for name, output in results["costs"].items()
    }
    if cost_totals:
        cost_df = pd.DataFrame(cost_totals)
        cost_df.index.name = "Year"
        cost_df = cost_df.reset_index()
        _write_chart_table(cost_df, "Operating Cost Summary", "column", subtype="stacked")

    cost_breakdown = pd.DataFrame(
        {
            "Category": list(cost_totals.keys()),
            "Amount": [float(series.sum()) for series in cost_totals.values()],
        }
    ) if cost_totals else pd.DataFrame({"Category": [], "Amount": []})
    if not cost_breakdown.empty:
        table_end = _write_table(
            writer,
            sheet,
            cost_breakdown,
            "Cost Breakdown",
            startrow=current_row,
            startcol=0,
            index=False,
        )
        pie = writer.book.add_chart({"type": "pie"})
        pie.add_series(
            {
                "name": "Cost Breakdown",
                "categories": [sheet, current_row + 2, 0, current_row + 1 + len(cost_breakdown.index), 0],
                "values": [sheet, current_row + 2, 1, current_row + 1 + len(cost_breakdown.index), 1],
            }
        )
        pie.set_title({"name": "Cost Breakdown"})
        worksheet.insert_chart(current_row, chart_col, pie, {"x_scale": 1.1, "y_scale": 1.1})
        current_row = max(table_end, current_row + chart_height)

    total_investment = _get_metric("Total Initial Investment", 0.0)
    debt_monthly = results["loan_schedule"].schedule.groupby("Month")["Closing Balance"].sum().sort_index()
    debt_annual = (
        debt_monthly.resample("Y").last().rename("Debt Closing Balance")
        if not debt_monthly.empty
        else pd.Series(dtype=float, name="Debt Closing Balance")
    )
    if debt_annual.empty:
        years = [projection.start_year]
        debt_values = [0.0]
    else:
        debt_annual.index = debt_annual.index.year
        years = debt_annual.index.tolist()
        debt_values = debt_annual.tolist()
    capex_values = [total_investment] + [0.0] * (len(years) - 1)
    capex_debt_df = pd.DataFrame(
        {
            "Year": years,
            "Capital Expenditure": capex_values,
            "Debt Closing Balance": debt_values,
        }
    )
    if not capex_debt_df.empty:
        _write_chart_table(capex_debt_df, "Capital Expenditure & Debt", "column")

    if not debt_monthly.empty:
        debt_schedule_df = debt_monthly.reset_index()
        debt_schedule_df["Month"] = debt_schedule_df["Month"].dt.to_period("M").astype(str)
        table_end = _write_table(
            writer,
            sheet,
            debt_schedule_df,
            "Debt Schedule",
            startrow=current_row,
            startcol=0,
            index=False,
        )
        debt_chart = writer.book.add_chart({"type": "line"})
        debt_chart.add_series(
            {
                "name": "Debt Balance",
                "categories": [sheet, current_row + 2, 0, current_row + 1 + len(debt_schedule_df.index), 0],
                "values": [sheet, current_row + 2, 1, current_row + 1 + len(debt_schedule_df.index), 1],
            }
        )
        debt_chart.set_title({"name": "Debt Schedule"})
        debt_chart.set_x_axis({"name": "Month"})
        debt_chart.set_y_axis({"name": "Balance"})
        worksheet.insert_chart(current_row, chart_col, debt_chart, {"x_scale": 1.1, "y_scale": 1.1})
        current_row = max(table_end, current_row + chart_height)


def _write_financial_performance(writer: pd.ExcelWriter, results: Dict[str, object]) -> None:
    sheet = "Financial Performance"
    income_monthly = results["financials"].income_monthly
    income_annual = results["financials"].income_annual
    total_expense = income_monthly[["COGS", "Staff Costs", "Other Opex", "Depreciation", "Interest", "Tax"]]
    next_row = _write_table(writer, sheet, income_monthly, "Monthly Financial Performance")
    next_row = _write_table(writer, sheet, income_annual, "Annual Financial Performance", startrow=next_row)
    _write_table(writer, sheet, total_expense, "Total Expense Schedule", startrow=next_row)


def _write_financial_position(writer: pd.ExcelWriter, results: Dict[str, object]) -> None:
    sheet = "Financial Position"
    balance_monthly = results["financials"].balance_monthly
    balance_annual = results["financials"].balance_annual
    next_row = _write_table(writer, sheet, balance_monthly, "Monthly Statement of Financial Position")
    _write_table(writer, sheet, balance_annual, "Annual Statement of Financial Position", startrow=next_row)


def _write_cash_flow_page(writer: pd.ExcelWriter, results: Dict[str, object]) -> None:
    sheet = "Cash Flow"
    cash_monthly = results["financials"].cashflow_monthly
    cash_annual = results["financials"].cashflow_annual
    next_row = _write_table(writer, sheet, cash_monthly, "Monthly Cash Flow Statement")
    next_row = _write_table(writer, sheet, cash_annual, "Annual Cash Flow Statement", startrow=next_row)
    cumulative = cash_monthly["Equity Cash Flow"].cumsum().to_frame("Cumulative Equity Cash Flow")
    _write_table(writer, sheet, cumulative, "Cumulative Equity Cash Flow", startrow=next_row)


def _write_sensitivity_page(
    writer: pd.ExcelWriter,
    model: CassavaBioethanolModel,
    scenarios: Iterable[SensitivityScenario],
) -> None:
    sheet = "Sensitivity Analyses"
    scenario_list = list(scenarios)
    config_df = pd.DataFrame([s.__dict__ for s in scenario_list]) if scenario_list else pd.DataFrame(columns=["name", "parameter", "delta"])
    next_row = _write_table(writer, sheet, config_df, "Sensitivity Analysis Configuration")
    if scenario_list:
        results = run_sensitivity(model, scenario_list)
    else:
        results = pd.DataFrame(columns=["Scenario", "Parameter", "Delta", "Project NPV", "Change vs Base"])
    next_row = _write_table(writer, sheet, results, "Sensitivity Results", startrow=next_row)

    mc_results = monte_carlo_simulation(
        model,
        parameter_std={"Corporate tax rate": 0.01, "Investor share capital": 0.02},
        iterations=200,
    )
    next_row = _write_table(writer, sheet, mc_results.describe().T, "Monte Carlo Simulation Results", startrow=next_row)
    tornado = tornado_chart_inputs(
        model,
        drivers=[("Corporate tax rate", 1.0), ("Investor share capital", 1.0), ("Owner share capital", 1.0)],
        scale=0.1,
    )
    _write_table(writer, sheet, tornado, "Tornado Chart Drivers", startrow=next_row)


def _write_scenario_page(
    writer: pd.ExcelWriter,
    model: CassavaBioethanolModel,
    configs: Iterable[ScenarioConfig],
) -> None:
    sheet = "Scenario Analysis"
    config_list = list(configs)
    config_df = pd.DataFrame([{"Scenario": cfg.name, **cfg.overrides} for cfg in config_list]) if config_list else pd.DataFrame()
    next_row = _write_table(writer, sheet, config_df, "Scenario/If Configuration")
    if config_list:
        comparison = scenario_comparison(model, config_list)
    else:
        comparison = pd.DataFrame(columns=["Scenario", "Project NPV", "Project IRR", "Equity IRR"])
    next_row = _write_table(writer, sheet, comparison, "Scenario Comparison", startrow=next_row)

    goal_seek_result = goal_seek_to_target(model, "Corporate tax rate", "Project NPV", comparison["Project NPV"].mean() if not comparison.empty else 0)
    goal_seek_df = pd.DataFrame([goal_seek_result.__dict__])
    _write_table(writer, sheet, goal_seek_df, "Goal Seek Results", startrow=next_row)


def _write_break_even_page(writer: pd.ExcelWriter, results: Dict[str, object]) -> None:
    sheet = "Break-even"
    break_even = results["break_even"]
    payback = results["payback"]
    next_row = _write_table(writer, sheet, break_even, "Break-even Analysis")
    _write_table(writer, sheet, payback, "Payback Schedule", startrow=next_row)
