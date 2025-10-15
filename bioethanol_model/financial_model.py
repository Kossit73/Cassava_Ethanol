from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

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

    def build(self) -> Dict[str, object]:
        projection = self.input_page.projection
        depreciation = compute_depreciation_schedule(
            self.input_page.initial_investment.data,
            projection.start_year,
            projection.end_year,
        )

        production = compute_production_tables(
            self.input_page.production_monthly.data,
            projection.start_year,
            projection.end_year,
        )

        revenue = compute_revenue_schedule(
            production,
            self.input_page.revenue_inputs.data,
            self.input_page.inflation_schedule.data,
        )

        cost_outputs = compute_cost_tables(
            self.input_page.direct_costs_monthly.data,
            self.input_page.staff_costs_monthly.data,
            self.input_page.other_opex_monthly.data,
            self.input_page.inflation_schedule.data,
        )

        loan_schedule = compute_loan_schedule(
            self.input_page.loan_schedule.data,
            projection.start_year,
            projection.end_year,
            self.input_page.initial_investment.data["Cost"].sum(),
        )

        ar_days = float(self.input_page.accounts_receivable.data.set_index("Metric").loc["Receivables days", "Value"])
        inventory_days = float(self.input_page.inventory_payable.data.set_index("Metric").loc["Inventory days", "Value"])
        ap_days = float(self.input_page.inventory_payable.data.set_index("Metric").loc["Payables days", "Value"])

        working_capital = compute_working_capital(
            revenue,
            cost_outputs,
            ar_days=ar_days,
            inventory_days=inventory_days,
            ap_days=ap_days,
        )

        tax_rate = float(
            self.input_page.global_inputs.data.set_index("Parameter").loc["Corporate tax rate", "Value"]
        )

        financials = compute_financial_statements(
            revenue,
            depreciation,
            cost_outputs,
            loan_schedule,
            working_capital,
            tax_rate=tax_rate,
        )

        metrics = compute_key_metrics(
            financials,
            discount_rate=float(
                self.input_page.global_inputs.data.set_index("Parameter").get("Discount rate", pd.Series([0.12]))[0]
            )
            if "Discount rate" in self.input_page.global_inputs.data["Parameter"].values
            else 0.12,
        )

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
        }
