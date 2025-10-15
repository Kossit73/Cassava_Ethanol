from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd

from . import inputs
from .schedules import (
    compute_break_even,
    compute_cost_tables,
    compute_depreciation_schedule,
    compute_financial_statements,
    compute_key_metrics,
    compute_loan_schedule,
    compute_payback,
    compute_production_tables,
    compute_revenue_schedule,
    compute_working_capital,
)
from .utils import irr, npv


@dataclass
class CassavaBioethanolModel:
    input_page: inputs.InputLandingPage = field(default_factory=inputs.default_input_page)
    scenario: str = "FARM_ONLY"

    SCENARIOS = ("FARM_ONLY", "BUY_ONLY", "HYBRID")

    def _prepare_page_for_scenario(self, scenario: str) -> inputs.InputLandingPage:
        page = copy.deepcopy(self.input_page)
        scenario = scenario.upper()
        global_inputs = page.global_inputs.data.copy()
        if not global_inputs.empty and "Parameter" in global_inputs.columns:
            lookup = global_inputs.set_index("Parameter")["Value"].to_dict()
        else:
            lookup = {}

        def _get_global(parameter: str, default: float) -> float:
            try:
                value = lookup.get(parameter, default)
                return float(value)
            except (TypeError, ValueError):
                return default

        farm_cost = _get_global("Cassava farm cost per ton", 45.0)
        purchase_cost = _get_global("Cassava purchase cost per ton", 70.0)
        farm_share = float(np.clip(_get_global("Hybrid farm share", 0.5), 0.0, 1.0))

        invest_df = page.initial_investment.data.copy()
        if not invest_df.empty and "Item" in invest_df.columns:
            farm_mask = invest_df["Item"].astype(str).str.contains("farm", case=False, na=False)
            numeric_costs = pd.to_numeric(invest_df.loc[farm_mask, "Cost"], errors="coerce").fillna(0.0)
            if scenario == "BUY_ONLY":
                invest_df.loc[farm_mask, "Cost"] = 0.0
            elif scenario == "HYBRID":
                invest_df.loc[farm_mask, "Cost"] = numeric_costs * farm_share
            else:
                invest_df.loc[farm_mask, "Cost"] = numeric_costs
            page.initial_investment.data = invest_df

        staff_df = page.staff_costs_monthly.data.copy()
        if not staff_df.empty and "Department" in staff_df.columns:
            farm_staff = staff_df["Department"].astype(str).str.contains("farm", case=False, na=False)
            if farm_staff.any():
                costs = pd.to_numeric(staff_df.loc[farm_staff, "Cost"], errors="coerce").fillna(0.0)
                heads = pd.to_numeric(staff_df.loc[farm_staff, "Headcount"], errors="coerce").fillna(0.0)
                if scenario == "BUY_ONLY":
                    staff_df.loc[farm_staff, "Cost"] = 0.0
                    staff_df.loc[farm_staff, "Headcount"] = 0.0
                elif scenario == "HYBRID":
                    staff_df.loc[farm_staff, "Cost"] = costs * farm_share
                    staff_df.loc[farm_staff, "Headcount"] = heads * farm_share
                else:
                    staff_df.loc[farm_staff, "Cost"] = costs
                    staff_df.loc[farm_staff, "Headcount"] = heads
                page.staff_costs_monthly.data = staff_df

        direct_df = page.direct_costs_monthly.data.copy()
        if not direct_df.empty and "Cost Category" in direct_df.columns:
            feed_mask = direct_df["Cost Category"].astype(str).str.contains("cassava", case=False, na=False)
            if feed_mask.any():
                prod_df = page.production_monthly.data.copy()
                tons = pd.Series(dtype=float)
                if not prod_df.empty and {"Month", "Cassava ton"}.issubset(prod_df.columns):
                    prod_df["Month"] = prod_df["Month"].astype(str)
                    tons = pd.to_numeric(prod_df.set_index("Month")["Cassava ton"], errors="coerce")
                default_ton = float(tons.mean()) if not tons.dropna().empty else 0.0

                def _tons(month: str) -> float:
                    if month in tons.index and pd.notna(tons.loc[month]):
                        return float(tons.loc[month])
                    return default_ton

                if scenario == "FARM_ONLY":
                    cost_per_ton = farm_cost
                elif scenario == "BUY_ONLY":
                    cost_per_ton = purchase_cost
                else:
                    cost_per_ton = farm_share * farm_cost + (1 - farm_share) * purchase_cost

                direct_df.loc[feed_mask, "Amount"] = direct_df.loc[feed_mask, "Month"].astype(str).apply(
                    lambda month: _tons(month) * cost_per_ton
                )
                page.direct_costs_monthly.data = direct_df

        return page

    def build(self, scenario: str | None = None) -> Dict[str, object]:
        scenario_name = (scenario or self.scenario or "FARM_ONLY").upper()
        if scenario_name not in self.SCENARIOS:
            raise ValueError(f"Unsupported scenario '{scenario_name}'. Expected one of {self.SCENARIOS}.")
        self.scenario = scenario_name

        page = self._prepare_page_for_scenario(scenario_name)

        projection = page.projection
        depreciation = compute_depreciation_schedule(
            page.initial_investment.data,
            projection.start_year,
            projection.end_year,
        )

        production = compute_production_tables(
            page.production_monthly.data,
            projection.start_year,
            projection.end_year,
        )

        revenue = compute_revenue_schedule(
            production,
            page.revenue_inputs.data,
            page.inflation_schedule.data,
        )

        cost_outputs = compute_cost_tables(
            page.direct_costs_monthly.data,
            page.staff_costs_monthly.data,
            page.other_opex_monthly.data,
            page.inflation_schedule.data,
        )

        loan_schedule = compute_loan_schedule(
            page.loan_schedule.data,
            projection.start_year,
            projection.end_year,
            page.initial_investment.data["Cost"].sum(),
        )

        receivables = page.accounts_receivable.data.set_index("Metric")
        inventory_inputs = page.inventory_payable.data.set_index("Metric")
        ar_days = float(receivables.loc["Receivables days", "Value"]) if "Receivables days" in receivables.index else 0.0
        inventory_days = (
            float(inventory_inputs.loc["Inventory days", "Value"]) if "Inventory days" in inventory_inputs.index else 0.0
        )
        ap_days = float(inventory_inputs.loc["Payables days", "Value"]) if "Payables days" in inventory_inputs.index else 0.0

        working_capital = compute_working_capital(
            revenue,
            cost_outputs,
            ar_days=ar_days,
            inventory_days=inventory_days,
            ap_days=ap_days,
        )

        global_inputs = page.global_inputs.data.set_index("Parameter")

        def _get_global(parameter: str, default: float) -> float:
            if parameter in global_inputs.index:
                try:
                    return float(global_inputs.loc[parameter, "Value"])
                except (TypeError, ValueError):
                    return default
            return default

        tax_rate = _get_global("Corporate tax rate", 0.28)

        financials = compute_financial_statements(
            revenue,
            depreciation,
            cost_outputs,
            loan_schedule,
            working_capital,
            tax_rate=tax_rate,
        )

        discount_rate = _get_global("Discount rate", 0.12)
        investor_share = _get_global("Investor share capital", 0.5)
        owner_share = _get_global("Owner share capital", max(0.0, 1 - investor_share))
        total_investment = float(page.initial_investment.data["Cost"].sum())

        metrics = compute_key_metrics(
            financials,
            discount_rate=discount_rate,
            total_investment=total_investment,
            investor_share=investor_share,
            owner_share=owner_share,
        )
        metrics.update(
            {
                "Corporate Tax Rate": tax_rate,
                "Investor Share": investor_share,
                "Owner Share": owner_share,
                "Terminal Growth Rate": _get_global("Terminal growth", 0.0),
                "Capital Gains Tax Rate": _get_global("Capital gains tax rate", 0.0),
                "Discount Rate": discount_rate,
                "Total Initial Investment": total_investment,
                "Scenario": scenario_name,
            }
        )
        if not np.isnan(metrics.get("Payback Period (months)", float("nan"))):
            metrics["Payback Period (years)"] = metrics["Payback Period (months)"] / 12.0

        break_even = compute_break_even(revenue, cost_outputs)
        payback = compute_payback(financials.cashflow_monthly)

        return {
            "depreciation": depreciation,
            "production": production,
            "revenue": revenue,
            "costs": cost_outputs,
            "loan_schedule": loan_schedule,
            "working_capital": working_capital,
            "financials": financials,
            "metrics": metrics,
            "break_even": break_even,
            "payback": payback,
            "scenario": scenario_name,
            "input_page_snapshot": page,
        }
