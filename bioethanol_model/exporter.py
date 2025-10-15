from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

from .financial_model import CassavaBioethanolModel
from .inputs import InputLandingPage
from .scenario import ScenarioConfig, goal_seek_to_target, scenario_comparison
from .sensitivity import SensitivityScenario, monte_carlo_simulation, run_sensitivity, tornado_chart_inputs


SECTION_GAP = 2


def _write_table(writer: pd.ExcelWriter, sheet: str, df: pd.DataFrame, title: str, startrow: int = 0) -> int:
    if sheet not in writer.sheets:
        worksheet = writer.book.add_worksheet(sheet)
        writer.sheets[sheet] = worksheet
    else:
        worksheet = writer.sheets[sheet]
    worksheet.write_string(startrow, 0, title)
    df_to_write = df.copy()
    df_to_write.to_excel(writer, sheet_name=sheet, startrow=startrow + 1, startcol=0)
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
        _write_key_metrics(writer, results)
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


def _write_key_metrics(writer: pd.ExcelWriter, results: Dict[str, object]) -> None:
    sheet = "Key Metrics"
    metrics = pd.Series(results["metrics"], name="Value").to_frame()
    _write_table(writer, sheet, metrics, "Assumptions Snapshot", startrow=0)

    global_summary = metrics.loc[["Project NPV", "Project IRR", "Equity IRR"]].copy()
    global_summary.rename(index={"Project NPV": "Project NPV", "Project IRR": "Project IRR", "Equity IRR": "Equity IRR"}, inplace=True)
    _write_table(writer, sheet, global_summary, "Overview", startrow=len(metrics) + SECTION_GAP + 2)

    latest = pd.DataFrame(
        {
            "Latest": {
                "Final Month Revenue": results["metrics"]["Final Month Revenue"],
                "Final Month EBITDA": results["metrics"]["Final Month EBITDA"],
                "Final Month Equity CF": results["metrics"]["Final Month Equity CF"],
                "Cumulative FCF": results["metrics"]["Cumulative FCF"],
                "Cumulative Equity CF": results["metrics"]["Cumulative Equity CF"],
            }
        }
    )
    _write_table(writer, sheet, latest, "Latest Drivers", startrow=metrics.shape[0] + global_summary.shape[0] + 3 * SECTION_GAP)

    production_chart = results["production"].annual
    production_chart.to_excel(writer, sheet_name=sheet, startrow=30, startcol=8)
    worksheet = writer.sheets[sheet]
    worksheet.write_string(29, 8, "Production Annual Summary")

    revenue_mix = results["revenue"].annual
    revenue_mix.to_excel(writer, sheet_name=sheet, startrow=30 + production_chart.shape[0] + SECTION_GAP, startcol=8)
    worksheet.write_string(29 + production_chart.shape[0] + SECTION_GAP, 8, "Revenue Mix")

    opex_mix = pd.concat(
        {name: output.annual.sum(axis=1) for name, output in results["costs"].items()}, axis=1
    )
    opex_mix.to_excel(writer, sheet_name=sheet, startrow=30, startcol=0)
    worksheet.write_string(29, 0, "Operating Cost Summary")

    debt_schedule = results["loan_schedule"].schedule
    debt_schedule.to_excel(writer, sheet_name=sheet, startrow=30 + opex_mix.shape[0] + SECTION_GAP, startcol=0)
    worksheet.write_string(29 + opex_mix.shape[0] + SECTION_GAP, 0, "Debt Schedule Detail")


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
