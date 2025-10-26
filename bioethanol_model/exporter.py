from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from .financial_model import CassavaBioethanolModel
from .inputs import InputLandingPage
from .scenario import ScenarioConfig, goal_seek_to_target, scenario_comparison
from .sensitivity import (
    DEFAULT_MONTE_CARLO_ITERATIONS,
    DEFAULT_MONTE_CARLO_SEED,
    SensitivityScenario,
    default_monte_carlo_parameters,
    monte_carlo_simulation,
    run_sensitivity,
    tornado_chart_inputs,
)
from .schedules import ExpenseSummary


SECTION_GAP = 2


def _reset_period_index(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Return a copy of *df* with its first index column renamed to *label*.

    Many of the model schedules keep year or month as the index. When we
    ``reset_index`` the resulting column name may vary (``index`` or the
    original index name). This helper normalises that behaviour so downstream
    selectors can always rely on the expected column label.
    """

    if df is None:
        return pd.DataFrame(columns=[label])

    reset = df.reset_index()
    if reset.empty:
        # Ensure the column still carries the desired heading even when empty.
        if reset.columns:
            reset = reset.rename(columns={reset.columns[0]: label})
        else:
            reset[label] = []
        return reset

    first_col = reset.columns[0]
    if first_col != label:
        reset = reset.rename(columns={first_col: label})
    return reset


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
    results: Dict[str, object] | None = None,
    scenario: str | None = None,
) -> Path:
    output_path = Path(output_path)
    if results is None:
        results = model.build(scenario=scenario)
    elif scenario is not None:
        model.scenario = scenario

    sensitivity_scenarios = list(sensitivity_scenarios or [])
    scenario_configs = list(scenario_configs or [])

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        scenario_page = results.get("input_page_snapshot", model.input_page)
        _write_input_page(writer, scenario_page)
        _write_financial_statements_page(writer, results)
        _write_key_metrics(writer, model, results)
        _write_financial_performance(writer, results)
        _write_financial_position(writer, results)
        _write_cash_flow_page(writer, results)
        _write_sensitivity_page(writer, model, results, sensitivity_scenarios)
        _write_scenario_page(writer, model, results, scenario_configs)
        _write_break_even_page(writer, model, results)
    return output_path


def _write_input_page(writer: pd.ExcelWriter, page: InputLandingPage) -> None:
    sheet = "Input Landing Page"
    if sheet in writer.sheets:
        worksheet = writer.sheets[sheet]
    else:
        worksheet = writer.book.add_worksheet(sheet)
        writer.sheets[sheet] = worksheet

    projection_df = page.projection.to_frame()
    next_row = _write_table(writer, sheet, projection_df, "Projection Horizon")

    heading_format = writer.book.add_format({"bold": True, "bg_color": "#F2F2F2"})

    for section, tables in page.grouped_tables().items():
        worksheet.write(next_row, 0, section, heading_format)
        next_row += 1
        for table in tables:
            next_row = _write_table(writer, sheet, table.model_frame, table.name, startrow=next_row)
            if table is page.initial_investment:
                worksheet.write(next_row, 0, "Total Initial Investment")
                worksheet.write_number(next_row, 1, page.total_initial_investment)
                next_row += SECTION_GAP + 1
        next_row += 1


def _write_financial_statements_page(writer: pd.ExcelWriter, results: Dict[str, object]) -> None:
    sheet = "Financial Statements"
    if sheet in writer.sheets:
        worksheet = writer.sheets[sheet]
    else:
        worksheet = writer.book.add_worksheet(sheet)
        writer.sheets[sheet] = worksheet

    expenses_summary: ExpenseSummary | None = (
        results.get("expenses") if isinstance(results.get("expenses"), ExpenseSummary) else None
    )

    statements = [
        (
            "Income Statement",
            _reset_period_index(results["financials"].income_monthly, "Month"),
            _reset_period_index(results["financials"].income_annual, "Year"),
        ),
        (
            "Cash Flow Statement",
            _reset_period_index(results["financials"].cashflow_monthly, "Month"),
            _reset_period_index(results["financials"].cashflow_annual, "Year"),
        ),
        (
            "Balance Sheet",
            _reset_period_index(results["financials"].balance_monthly, "Month"),
            _reset_period_index(results["financials"].balance_annual, "Year"),
        ),
    ]

    heading_format = writer.book.add_format({"bold": True, "bg_color": "#F2F2F2"})

    current_row = 0
    for title, monthly, annual in statements:
        worksheet.write(current_row, 0, title, heading_format)
        current_row += 1
        current_row = _write_table(
            writer,
            sheet,
            monthly,
            f"{title} (Monthly)",
            startrow=current_row,
            index=False,
        )
        current_row = _write_table(
            writer,
            sheet,
            annual,
            f"{title} (Annual)",
            startrow=current_row,
            index=False,
        )
        current_row += 1

        if title == "Income Statement" and isinstance(expenses_summary, ExpenseSummary):
            expense_monthly = expenses_summary.monthly
            expense_annual = expenses_summary.annual

            if isinstance(expense_monthly, pd.DataFrame) and not expense_monthly.empty:
                expense_monthly_tbl = _reset_period_index(expense_monthly, "Month")
                current_row = _write_table(
                    writer,
                    sheet,
                    expense_monthly_tbl,
                    "Income Statement Expense Breakdown (Monthly)",
                    startrow=current_row,
                    index=False,
                )

            if isinstance(expense_annual, pd.DataFrame) and not expense_annual.empty:
                expense_annual_tbl = _reset_period_index(expense_annual, "Year")
                current_row = _write_table(
                    writer,
                    sheet,
                    expense_annual_tbl,
                    "Income Statement Expense Breakdown (Annual)",
                    startrow=current_row,
                    index=False,
                )


def _write_key_metrics(writer: pd.ExcelWriter, model: CassavaBioethanolModel, results: Dict[str, object]) -> None:
    sheet = "Key Metrics"
    if sheet in writer.sheets:
        worksheet = writer.sheets[sheet]
    else:
        worksheet = writer.book.add_worksheet(sheet)
        writer.sheets[sheet] = worksheet

    metrics = results["metrics"]
    projection = model.input_page.projection
    expenses: ExpenseSummary | None = results.get("expenses")  # type: ignore[assignment]

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
                projection.planning_start,
                projection.end_year - projection.start_year + 1,
                _get_metric("Discount Rate"),
                _get_metric("Terminal Growth Rate"),
                _get_metric("Capital Gains Tax Rate"),
                _get_metric("Total Initial Investment"),
                _get_metric("Initial Equity Investment"),
                _get_metric("Initial Loan Funding"),
            ]
        },
        index=[
            "Start Year",
            "End Year",
            "Planning Start Month",
            "Projection Years",
            "Discount Rate",
            "Terminal Growth Rate",
            "Capital Gains Tax Rate",
            "Total Initial Investment",
            "Initial Equity Investment",
            "Initial Loan Funding",
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
    )
    annual_ops = _reset_period_index(annual_ops, "Year")
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
        numeric_cols = [
            col_idx
            for col_idx in cols
            if col_idx != categories_col and np.issubdtype(data.iloc[:, col_idx].dtype, np.number)
        ]
        if not numeric_cols:
            current_row = max(table_end, current_row + chart_height)
            return
        for col_idx in numeric_cols:
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
            for col in [
                "Operating Cash Flow",
                "Investing Cash Flow",
                "Financing Cash Flow",
                "Free Cash Flow",
                "Equity Cash Flow",
            ]
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

    production_df = _reset_period_index(results["production"].annual, "Year")
    if not production_df.empty:
        _write_chart_table(production_df, "Annual Production", "line")

    cashflow_annual = _reset_period_index(results["financials"].cashflow_annual, "Year")
    cash_columns = [
        "Operating Cash Flow",
        "Investing Cash Flow",
        "Financing Cash Flow",
        "Free Cash Flow",
        "Equity Cash Flow",
    ]
    cash_columns = [c for c in cash_columns if c in cashflow_annual.columns]
    if cash_columns:
        cash_df = cashflow_annual[["Year", *cash_columns]]
        _write_chart_table(cash_df, "Cash Flow Summary", "column")

    revenue_df = _reset_period_index(results["revenue"].annual, "Year")
    if not revenue_df.empty:
        exclude = ["Total Revenue"] if "Total Revenue" in revenue_df.columns else None
        _write_chart_table(
            revenue_df,
            "Revenue Mix",
            "column",
            subtype="stacked",
            exclude_columns=exclude,
        )

    expense_annual = pd.DataFrame()
    if isinstance(expenses, ExpenseSummary):
        expense_annual = expenses.annual

    if not expense_annual.empty:
        cost_totals = {col: expense_annual[col] for col in expense_annual.columns}
    else:
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
    expenses_summary = results.get("expenses") if isinstance(results.get("expenses"), ExpenseSummary) else None

    expense_cols = ["COGS", "Staff Costs", "Other Opex", "Tax"]
    primary_expense_cols = ["COGS", "Staff Costs", "Other Opex"]
    monthly_expense = pd.DataFrame(index=income_monthly.index)
    annual_expense = pd.DataFrame(index=income_annual.index)

    if isinstance(expenses_summary, ExpenseSummary):
        if not expenses_summary.monthly.empty:
            monthly_expense = expenses_summary.monthly.reindex(
                columns=[col for col in expense_cols if col in expenses_summary.monthly.columns]
            )
        if not expenses_summary.annual.empty:
            annual_expense = expenses_summary.annual.reindex(
                columns=[col for col in expense_cols if col in expenses_summary.annual.columns]
            )

    for col in primary_expense_cols:
        if col not in monthly_expense.columns:
            if col in income_monthly.columns:
                monthly_expense[col] = income_monthly[col]
            else:
                monthly_expense[col] = 0.0
        if col not in annual_expense.columns:
            if col in income_annual.columns:
                annual_expense[col] = income_annual[col]
            else:
                annual_expense[col] = 0.0

    for extra in ("Depreciation", "Interest"):
        if extra in income_monthly.columns:
            monthly_expense[extra] = income_monthly[extra]
        if extra in income_annual.columns:
            annual_expense[extra] = income_annual[extra]

    ordered_cols = [col for col in expense_cols if col in monthly_expense.columns]
    for extra in ("Depreciation", "Interest"):
        if extra in monthly_expense.columns:
            ordered_cols.append(extra)
    ordered_cols = list(dict.fromkeys(ordered_cols))
    monthly_expense = monthly_expense.reindex(columns=ordered_cols)
    monthly_expense = monthly_expense.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if not monthly_expense.empty:
        monthly_expense["Total"] = monthly_expense.sum(axis=1)

    annual_ordered_cols = [col for col in expense_cols if col in annual_expense.columns]
    for extra in ("Depreciation", "Interest"):
        if extra in annual_expense.columns:
            annual_ordered_cols.append(extra)
    annual_ordered_cols = list(dict.fromkeys(annual_ordered_cols))
    annual_expense = annual_expense.reindex(columns=annual_ordered_cols)
    annual_expense = annual_expense.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if not annual_expense.empty:
        annual_expense["Total"] = annual_expense.sum(axis=1)

    next_row = _write_table(writer, sheet, income_monthly, "Monthly Financial Performance")
    next_row = _write_table(writer, sheet, income_annual, "Annual Financial Performance", startrow=next_row)
    next_row = _write_table(writer, sheet, monthly_expense, "Expense Breakdown (Monthly)", startrow=next_row)
    if not annual_expense.empty:
        next_row = _write_table(writer, sheet, annual_expense, "Expense Breakdown (Annual)", startrow=next_row)

    income_ratios_monthly = getattr(results["financials"], "income_ratios_monthly", pd.DataFrame())
    income_ratios_annual = getattr(results["financials"], "income_ratios_annual", pd.DataFrame())
    if isinstance(income_ratios_monthly, pd.DataFrame) and not income_ratios_monthly.empty:
        ratio_monthly = _reset_period_index(income_ratios_monthly, "Month")
        if "Month" in ratio_monthly.columns:
            try:
                ratio_monthly["Month"] = pd.to_datetime(ratio_monthly["Month"]).dt.to_period("M").astype(str)
            except Exception:
                ratio_monthly["Month"] = ratio_monthly["Month"].astype(str)
        next_row = _write_table(
            writer,
            sheet,
            ratio_monthly,
            "Income Statement Ratios (Monthly)",
            startrow=next_row,
            index=False,
        )
    if isinstance(income_ratios_annual, pd.DataFrame) and not income_ratios_annual.empty:
        ratio_annual = _reset_period_index(income_ratios_annual, "Year")
        next_row = _write_table(
            writer,
            sheet,
            ratio_annual,
            "Income Statement Ratios (Annual)",
            startrow=next_row,
            index=False,
        )

    staff_schedule = results.get("staff_schedule")
    if staff_schedule is not None:
        positions_df = getattr(staff_schedule, "positions", pd.DataFrame())
        summary_df = getattr(staff_schedule, "department_summary", pd.DataFrame())
        if isinstance(positions_df, pd.DataFrame) and not positions_df.empty:
            next_row = _write_table(writer, sheet, positions_df, "Staff Position Schedule", startrow=next_row)
        if isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
            _write_table(writer, sheet, summary_df, "Staff Cost by Department", startrow=next_row)


def _write_financial_position(writer: pd.ExcelWriter, results: Dict[str, object]) -> None:
    sheet = "Financial Position"
    balance_monthly = results["financials"].balance_monthly
    balance_annual = results["financials"].balance_annual
    next_row = _write_table(writer, sheet, balance_monthly, "Monthly Statement of Financial Position")
    next_row = _write_table(writer, sheet, balance_annual, "Annual Statement of Financial Position", startrow=next_row)

    balance_ratios_monthly = getattr(results["financials"], "balance_ratios_monthly", pd.DataFrame())
    balance_ratios_annual = getattr(results["financials"], "balance_ratios_annual", pd.DataFrame())
    if isinstance(balance_ratios_monthly, pd.DataFrame) and not balance_ratios_monthly.empty:
        ratio_monthly = _reset_period_index(balance_ratios_monthly, "Month")
        if "Month" in ratio_monthly.columns:
            try:
                ratio_monthly["Month"] = pd.to_datetime(ratio_monthly["Month"]).dt.to_period("M").astype(str)
            except Exception:
                ratio_monthly["Month"] = ratio_monthly["Month"].astype(str)
        next_row = _write_table(
            writer,
            sheet,
            ratio_monthly,
            "Statement of Financial Position Ratios (Monthly)",
            startrow=next_row,
            index=False,
        )
    if isinstance(balance_ratios_annual, pd.DataFrame) and not balance_ratios_annual.empty:
        ratio_annual = _reset_period_index(balance_ratios_annual, "Year")
        _write_table(
            writer,
            sheet,
            ratio_annual,
            "Statement of Financial Position Ratios (Annual)",
            startrow=next_row,
            index=False,
        )


def _write_cash_flow_page(writer: pd.ExcelWriter, results: Dict[str, object]) -> None:
    sheet = "Cash Flow"
    if sheet in writer.sheets:
        worksheet = writer.sheets[sheet]
    else:
        worksheet = writer.book.add_worksheet(sheet)
        writer.sheets[sheet] = worksheet

    chart_col = 8
    chart_height = 18

    cash_monthly = results["financials"].cashflow_monthly
    cash_annual = results["financials"].cashflow_annual

    def _format_month_table(df: pd.DataFrame) -> pd.DataFrame:
        table = _reset_period_index(df, "Month")
        if "Month" in table.columns:
            try:
                table["Month"] = pd.to_datetime(table["Month"]).dt.to_period("M").astype(str)
            except Exception:
                table["Month"] = table["Month"].astype(str)
        return table

    monthly_table = _format_month_table(cash_monthly)
    monthly_start = 0
    next_row = _write_table(
        writer,
        sheet,
        monthly_table,
        "Monthly Cash Flow Statement",
        startrow=monthly_start,
        index=False,
    )

    cash_columns = [c for c in monthly_table.columns if c != "Month"]
    if cash_columns and len(monthly_table.index) > 0:
        chart = writer.book.add_chart({"type": "column"})
        header_row = monthly_start + 1
        data_start = monthly_start + 2
        data_end = monthly_start + 1 + len(monthly_table.index)
        for idx, column in enumerate(cash_columns, start=1):
            chart.add_series(
                {
                    "name": [sheet, header_row, idx],
                    "categories": [sheet, data_start, 0, data_end, 0],
                    "values": [sheet, data_start, idx, data_end, idx],
                }
            )
        chart.set_title({"name": "Cash Flow & Returns"})
        chart.set_x_axis({"name": "Month"})
        chart.set_y_axis({"major_gridlines": {"visible": True}})
        chart.set_legend({"position": "bottom"})
        worksheet.insert_chart(monthly_start, chart_col, chart, {"x_scale": 1.1, "y_scale": 1.1})
        next_row = max(next_row, monthly_start + chart_height)

    annual_table = _reset_period_index(cash_annual, "Year")
    annual_start = next_row
    next_row = _write_table(
        writer,
        sheet,
        annual_table,
        "Annual Cash Flow Statement",
        startrow=annual_start,
        index=False,
    )

    cumulative_series = {}
    if "Free Cash Flow" in cash_monthly.columns:
        cumulative_series["Cumulative Free Cash Flow"] = cash_monthly["Free Cash Flow"].cumsum()
    if "Equity Cash Flow" in cash_monthly.columns:
        cumulative_series["Cumulative Equity Cash Flow"] = cash_monthly["Equity Cash Flow"].cumsum()

    if cumulative_series and len(monthly_table.index) > 0:
        cumulative_df = pd.DataFrame({"Month": monthly_table["Month"]})
        for name, series in cumulative_series.items():
            cumulative_df[name] = series.values
        cumulative_start = next_row
        next_row = _write_table(
            writer,
            sheet,
            cumulative_df,
            "Cumulative Equity Cash Flow Schedule",
            startrow=cumulative_start,
            index=False,
        )

        chart = writer.book.add_chart({"type": "line"})
        header_row = cumulative_start + 1
        data_start = cumulative_start + 2
        data_end = cumulative_start + 1 + len(cumulative_df.index)
        for idx, column in enumerate(cumulative_df.columns[1:], start=1):
            chart.add_series(
                {
                    "name": [sheet, header_row, idx],
                    "categories": [sheet, data_start, 0, data_end, 0],
                    "values": [sheet, data_start, idx, data_end, idx],
                }
            )
        chart.set_title({"name": "Cumulative Cash Flows"})
        chart.set_x_axis({"name": "Month"})
        chart.set_y_axis({"major_gridlines": {"visible": True}})
        chart.set_legend({"position": "bottom"})
        worksheet.insert_chart(cumulative_start, chart_col, chart, {"x_scale": 1.1, "y_scale": 1.1})
        next_row = max(next_row, cumulative_start + chart_height)


def _write_sensitivity_page(
    writer: pd.ExcelWriter,
    model: CassavaBioethanolModel,
    results: Dict[str, object],
    scenarios: Iterable[SensitivityScenario],
) -> None:
    sheet = "Sensitivity Analyses"
    scenario_list = list(scenarios)
    config_df = (
        pd.DataFrame([s.__dict__ for s in scenario_list])
        if scenario_list
        else pd.DataFrame(columns=["name", "parameter", "delta"])
    )
    next_row = _write_table(writer, sheet, config_df, "Sensitivity Analysis Configuration")
    base_page = copy.deepcopy(results.get("input_page_snapshot", model.input_page))

    def _scenario_model() -> CassavaBioethanolModel:
        clone = CassavaBioethanolModel(copy.deepcopy(base_page))
        clone.scenario = model.scenario
        return clone

    if scenario_list:
        sensitivity_model = _scenario_model()
        sensitivity_results = run_sensitivity(sensitivity_model, scenario_list)
    else:
        sensitivity_results = pd.DataFrame(columns=["Scenario", "Parameter", "Delta", "Project NPV", "Change vs Base"])
    next_row = _write_table(writer, sheet, sensitivity_results, "Simulation Results", startrow=next_row)

    mc_iterations = DEFAULT_MONTE_CARLO_ITERATIONS
    mc_seed = DEFAULT_MONTE_CARLO_SEED
    mc_params = default_monte_carlo_parameters()

    settings_df = pd.DataFrame(
        [
            {"Setting": "Iterations", "Value": mc_iterations},
            {"Setting": "Random Seed", "Value": mc_seed},
        ]
    )
    next_row = _write_table(
        writer,
        sheet,
        settings_df,
        "Monte Carlo Simulation Settings",
        startrow=next_row,
        index=False,
    )
    next_row = _write_table(
        writer,
        sheet,
        mc_params,
        "Monte Carlo Parameter Configuration",
        startrow=next_row,
        index=False,
    )

    mc_model = _scenario_model()
    mc_results = monte_carlo_simulation(
        mc_model,
        parameter_configs=mc_params,
        iterations=mc_iterations,
        random_seed=mc_seed,
    )
    next_row = _write_table(
        writer,
        sheet,
        mc_results.describe().T,
        "Monte Carlo Simulation Results",
        startrow=next_row,
    )
    tornado_model = _scenario_model()
    tornado = tornado_chart_inputs(
        tornado_model,
        drivers=[("Corporate tax rate", 1.0), ("Investor share capital", 1.0), ("Owner share capital", 1.0)],
        scale=0.1,
    )
    _write_table(writer, sheet, tornado, "Tornado Chart Drivers", startrow=next_row)


def _write_scenario_page(
    writer: pd.ExcelWriter,
    model: CassavaBioethanolModel,
    base_results: Dict[str, object],
    configs: Iterable[ScenarioConfig],
) -> None:
    sheet = "Scenario Analysis"

    config_list = list(configs)
    config_df = (
        pd.DataFrame([{"Scenario": cfg.name, **cfg.overrides} for cfg in config_list])
        if config_list
        else pd.DataFrame()
    )
    next_row = _write_table(writer, sheet, config_df, "Scenario/Is Configuration", index=False)

    base_page = copy.deepcopy(base_results.get("input_page_snapshot", model.input_page))
    base_inputs = base_page.global_inputs.model_frame
    tool_df = base_inputs.rename(columns={"Value": "Base Value"})
    numeric_values = pd.to_numeric(tool_df["Base Value"], errors="coerce")
    tool_df["Low Bound"] = np.where(numeric_values.notna(), numeric_values * 0.8, np.nan)
    tool_df["High Bound"] = np.where(numeric_values.notna(), numeric_values * 1.2, np.nan)
    desired_order = ["Parameter", "Base Value", "Units", "Low Bound", "High Bound"]
    tool_df = tool_df[[c for c in desired_order if c in tool_df.columns]]
    next_row = _write_table(
        writer,
        sheet,
        tool_df,
        "Scenario Tool Configuration",
        startrow=next_row,
        index=False,
    )

    def _scenario_model() -> CassavaBioethanolModel:
        clone = CassavaBioethanolModel(copy.deepcopy(base_page))
        clone.scenario = model.scenario
        return clone

    if config_list:
        comparison_model = _scenario_model()
        comparison = scenario_comparison(comparison_model, config_list)
    else:
        comparison = pd.DataFrame(columns=["Scenario", "Project NPV", "Project IRR", "Equity IRR"])
    next_row = _write_table(writer, sheet, comparison, "Scenario Comparison", startrow=next_row, index=False)

    base_metrics = base_results.get("metrics", {})
    goal_seek_parameter = "Corporate tax rate"
    goal_seek_metric = "Project NPV"
    target_value = (
        comparison[goal_seek_metric].mean()
        if not comparison.empty and goal_seek_metric in comparison
        else float(base_metrics.get(goal_seek_metric, 0.0))
    )

    goal_seek_config = pd.DataFrame(
        {
            "Parameter": [goal_seek_parameter],
            "Target Metric": [goal_seek_metric],
            "Target Value": [target_value],
        }
    )
    next_row = _write_table(
        writer,
        sheet,
        goal_seek_config,
        "Goal Seek Configuration",
        startrow=next_row,
        index=False,
    )

    try:
        goal_model = _scenario_model()
        goal_seek_result = goal_seek_to_target(
            goal_model,
            goal_seek_parameter,
            goal_seek_metric,
            target_value,
        )
        goal_seek_df = pd.DataFrame(
            [
                {
                    "Target Name": goal_seek_result.target_name,
                    "Achieved Value": goal_seek_result.achieved_value,
                    "Tolerance": goal_seek_result.tolerance,
                    "Iterations": goal_seek_result.iterations,
                }
            ]
        )
        goal_seek_df.insert(0, "Target Metric", goal_seek_metric)
        goal_seek_df.insert(0, "Parameter", goal_seek_parameter)
        goal_seek_df["Target Value"] = target_value
    except KeyError:
        goal_seek_df = pd.DataFrame(
            columns=["Parameter", "Target Metric", "Target Value", "Target Name", "Achieved Value", "Tolerance", "Iterations"]
        )
    _write_table(writer, sheet, goal_seek_df, "Goal Seek Results", startrow=next_row, index=False)


def _write_break_even_page(
    writer: pd.ExcelWriter,
    model: CassavaBioethanolModel,
    results: Dict[str, object],
) -> None:
    sheet = "Break-even"

    production_monthly = results["production"].monthly
    revenue_monthly = results["revenue"].monthly
    direct_costs = results["costs"].get("Direct Costs")
    staff_costs = results["costs"].get("Staff Costs")
    other_costs = results["costs"].get("Other Opex")
    revenue_inputs = model.input_page.revenue_inputs.model_frame

    if not production_monthly.empty:
        if "Ethanol litres" in production_monthly.columns:
            volume_series = production_monthly["Ethanol litres"]
        else:
            numeric_cols = production_monthly.select_dtypes(include=[np.number]).columns
            volume_series = production_monthly[numeric_cols[0]] if len(numeric_cols) else pd.Series(dtype=float)
    else:
        volume_series = pd.Series(dtype=float)

    total_volume = float(volume_series.sum()) if not volume_series.empty else float("nan")
    total_revenue = float(revenue_monthly.get("Total Revenue", pd.Series(dtype=float)).sum())
    total_direct = float(direct_costs.monthly.sum().sum()) if direct_costs else 0.0
    fixed_total = 0.0
    if staff_costs:
        fixed_total += float(staff_costs.monthly.sum().sum())
    if other_costs:
        fixed_total += float(other_costs.monthly.sum().sum())

    avg_price = (
        total_revenue / total_volume
        if np.isfinite(total_volume) and total_volume != 0
        else float("nan")
    )
    avg_variable_cost = (
        total_direct / total_volume
        if np.isfinite(total_volume) and total_volume != 0
        else float("nan")
    )
    contribution = (
        avg_price - avg_variable_cost
        if np.isfinite(avg_price) and np.isfinite(avg_variable_cost)
        else float("nan")
    )
    break_even_volume = (
        fixed_total / contribution
        if np.isfinite(contribution) and contribution != 0
        else float("nan")
    )

    if "Base Price" in revenue_inputs and not revenue_inputs.empty:
        try:
            base_price_input = float(revenue_inputs["Base Price"].iloc[0])
        except (TypeError, ValueError):
            base_price_input = float("nan")
    else:
        base_price_input = float("nan")

    break_even_input = pd.DataFrame(
        {
            "Metric": [
                "Base Price (input)",
                "Average Selling Price (per unit)",
                "Average Variable Cost (per unit)",
                "Annual Fixed Costs",
                "Total Production Volume",
                "Break-even Volume",
            ],
            "Value": [base_price_input, avg_price, avg_variable_cost, fixed_total, total_volume, break_even_volume],
        }
    )

    break_even_df = _reset_period_index(results["break_even"], "Month")
    if "Month" in break_even_df.columns:
        try:
            break_even_df["Month"] = pd.to_datetime(break_even_df["Month"]).dt.to_period("M").astype(str)
        except Exception:
            break_even_df["Month"] = break_even_df["Month"].astype(str)
    if "Break-even Month" in break_even_df.columns:
        break_even_df["Break-even Month"] = break_even_df["Break-even Month"].astype(str)

    payback_df = _reset_period_index(results["payback"], "Month")
    if "Month" in payback_df.columns:
        try:
            payback_df["Month"] = pd.to_datetime(payback_df["Month"]).dt.to_period("M").astype(str)
        except Exception:
            payback_df["Month"] = payback_df["Month"].astype(str)
    if "Payback Month" in payback_df.columns:
        payback_df["Payback Month"] = payback_df["Payback Month"].astype(str)

    next_row = _write_table(writer, sheet, break_even_input, "Break-Even Analysis Input", index=False)
    next_row = _write_table(writer, sheet, break_even_df, "Break-Even Results", startrow=next_row, index=False)
    _write_table(writer, sheet, payback_df, "Payback Schedule", startrow=next_row, index=False)
