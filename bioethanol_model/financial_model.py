from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple, TYPE_CHECKING

import hashlib
import numpy as np
import pandas as pd

from . import inputs
if TYPE_CHECKING:
    from .advanced_tools import AdvancedAnalyticsToolkit

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
    compute_staff_schedule,
    compute_working_capital,
    extract_expense_summary,
    ExpenseSummary,
)
from .utils import irr, npv


@dataclass
class CassavaBioethanolModel:
    input_page: inputs.InputLandingPage = field(default_factory=inputs.default_input_page)
    scenario: str = "FARM_ONLY"
    _scenario_cache: Dict[str, Tuple[str, Dict[str, object]]] = field(default_factory=dict, init=False, repr=False)
    _advanced_tools: "AdvancedAnalyticsToolkit" | None = field(default=None, init=False, repr=False)

    SCENARIOS = ("FARM_ONLY", "BUY_ONLY", "HYBRID")

    @classmethod
    def default(cls) -> "CassavaBioethanolModel":
        """Return a model seeded with the default input landing page."""

        return cls()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _hash_dataframe(self, df: pd.DataFrame | None) -> str:
        if df is None or getattr(df, "empty", True):
            return "empty"
        normalised = df.copy()
        normalised.index = normalised.index.astype(str)
        normalised = normalised.fillna(0)
        return hashlib.sha1(normalised.to_csv().encode("utf-8")).hexdigest()

    def _input_signature(self) -> str:
        return self.input_page.signature()

    def _result_signature(self, result: Dict[str, object]) -> str:
        financials = result.get("financials")
        if financials is None:
            return ""
        parts = [
            self._hash_dataframe(getattr(financials, "income_monthly", None)),
            self._hash_dataframe(getattr(financials, "cashflow_monthly", None)),
            self._hash_dataframe(getattr(financials, "balance_monthly", None)),
        ]
        return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()

    def _prepare_page_for_scenario(self, scenario: str) -> inputs.InputLandingPage:
        page = copy.deepcopy(self.input_page)
        scenario = scenario.upper()
        global_inputs = page.global_inputs.model_frame
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

        farm_cost = _get_global("Cassava farm cost per ton", 0.0)
        purchase_cost = _get_global("Cassava purchase cost per ton", 0.0)
        farm_share = float(np.clip(_get_global("Hybrid farm share", 0.0), 0.0, 1.0))

        invest_df = page.initial_investment.model_frame
        if not invest_df.empty and "Item" in invest_df.columns:
            farm_mask = invest_df["Item"].astype(str).str.contains("farm", case=False, na=False)
            numeric_costs = pd.to_numeric(invest_df.loc[farm_mask, "Cost"], errors="coerce").fillna(0.0)
            if scenario == "BUY_ONLY":
                invest_df.loc[farm_mask, "Cost"] = 0.0
            elif scenario == "HYBRID":
                invest_df.loc[farm_mask, "Cost"] = numeric_costs * farm_share
            else:
                invest_df.loc[farm_mask, "Cost"] = numeric_costs
            if not invest_df.equals(page.initial_investment.data):
                mark_user = page.initial_investment.placeholder
                page.initial_investment.set_data(invest_df, mark_user_input=mark_user)

        staff_df = page.staff_costs_monthly.model_frame
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
                mark_user = page.staff_costs_monthly.placeholder or farm_staff.any()
                page.staff_costs_monthly.set_data(staff_df, mark_user_input=mark_user)

        positions_df = page.staff_positions.model_frame
        if not positions_df.empty and "Department" in positions_df.columns:
            farm_positions = positions_df["Department"].astype(str).str.contains("farm", case=False, na=False)
            if farm_positions.any():
                heads = pd.to_numeric(positions_df.loc[farm_positions, "Headcount"], errors="coerce").fillna(0.0)
                if scenario == "BUY_ONLY":
                    positions_df.loc[farm_positions, "Headcount"] = 0.0
                elif scenario == "HYBRID":
                    positions_df.loc[farm_positions, "Headcount"] = heads * farm_share
                else:
                    positions_df.loc[farm_positions, "Headcount"] = heads
                mark_user = page.staff_positions.placeholder or farm_positions.any()
                page.staff_positions.set_data(positions_df, mark_user_input=mark_user)

        return page

    def _apply_staff_schedule(self, page: inputs.InputLandingPage):
        """Update monthly staff costs from the staff position salary schedule."""

        schedule = compute_staff_schedule(page.staff_positions.model_frame)

        staff_df = page.staff_costs_monthly.model_frame
        if not staff_df.empty and "Department" in staff_df.columns:
            dept_salary = {}
            summary = schedule.department_summary
            if not summary.empty and "Average Monthly Salary" in summary.columns:
                dept_salary = summary.set_index("Department")["Average Monthly Salary"].to_dict()

            staff_df["Headcount"] = pd.to_numeric(staff_df["Headcount"], errors="coerce").fillna(0.0)
            updated_costs = []
            for _, row in staff_df.iterrows():
                dept = row.get("Department")
                headcount = float(row.get("Headcount", 0.0) or 0.0)
                salary = dept_salary.get(dept)
                if salary is None or not np.isfinite(salary):
                    try:
                        current_cost = float(row.get("Cost", 0.0))
                    except (TypeError, ValueError):
                        current_cost = 0.0
                    updated_costs.append(current_cost)
                else:
                    updated_costs.append(headcount * salary)
            staff_df["Cost"] = updated_costs
            mark_user = page.staff_costs_monthly.placeholder or bool(dept_salary)
            page.staff_costs_monthly.set_data(staff_df, mark_user_input=mark_user)

        return schedule

    # ------------------------------------------------------------------
    # Advanced analytics extensions
    # ------------------------------------------------------------------

    def advanced_toolkit(self) -> "AdvancedAnalyticsToolkit":
        """Lazily instantiate the :class:`AdvancedAnalyticsToolkit` helper."""

        if self._advanced_tools is None:
            from .advanced_tools import AdvancedAnalyticsToolkit

            self._advanced_tools = AdvancedAnalyticsToolkit(self)
        return self._advanced_tools

    def build(self, scenario: str | None = None) -> Dict[str, object]:
        scenario_name = (scenario or self.scenario or "FARM_ONLY").upper()
        if scenario_name not in self.SCENARIOS:
            raise ValueError(f"Unsupported scenario '{scenario_name}'. Expected one of {self.SCENARIOS}.")
        self.scenario = scenario_name

        signature = self._input_signature()
        cached = self._scenario_cache.get(scenario_name)
        if cached and cached[0] == signature:
            return copy.deepcopy(cached[1])

        page = self._prepare_page_for_scenario(scenario_name)

        staff_schedule = self._apply_staff_schedule(page)

        projection = page.projection
        depreciation = compute_depreciation_schedule(
            page.initial_investment.model_frame,
            projection.start_year,
            projection.end_year,
        )

        planning_start = projection.planning_start_timestamp

        production = compute_production_tables(
            page.production_monthly.model_frame,
            projection.start_year,
            projection.end_year,
            planning_start=planning_start,
        )

        revenue = compute_revenue_schedule(
            production,
            page.revenue_inputs.model_frame,
            page.inflation_schedule.model_frame,
            planning_start=planning_start,
        )

        cost_outputs = compute_cost_tables(
            page.direct_costs_monthly.model_frame,
            page.staff_costs_monthly.model_frame,
            page.other_opex_monthly.model_frame,
            page.inflation_schedule.model_frame,
            projection.start_year,
            projection.end_year,
        )

        loan_schedule = compute_loan_schedule(
            page.loan_schedule.model_frame,
            projection.start_year,
            projection.end_year,
        )

        working_capital = compute_working_capital(
            revenue,
            cost_outputs,
            page.accounts_receivable.model_frame,
            page.inventory_payable.model_frame,
        )

        global_inputs = page.global_inputs.model_frame.set_index("Parameter")

        def _get_global(parameter: str, default: float) -> float:
            if parameter in global_inputs.index:
                try:
                    return float(global_inputs.loc[parameter, "Value"])
                except (TypeError, ValueError):
                    return default
            return default

        tax_rate = _get_global("Corporate tax rate", 0.0)

        financials = compute_financial_statements(
            revenue,
            depreciation,
            cost_outputs,
            loan_schedule,
            working_capital,
            tax_rate=tax_rate,
        )

        expenses: ExpenseSummary = extract_expense_summary(financials, cost_outputs)

        discount_rate = _get_global("Discount rate", 0.0)
        investor_share = _get_global("Investor share capital", 0.0)
        owner_share = _get_global("Owner share capital", float("nan"))
        if not np.isfinite(owner_share):
            owner_share = max(0.0, 1.0 - investor_share)
        init_df = page.initial_investment.model_frame
        total_investment = float(init_df["Cost"].sum()) if "Cost" in init_df.columns else 0.0

        metrics = compute_key_metrics(
            financials,
            discount_rate=discount_rate,
            investor_share=investor_share,
            owner_share=owner_share,
            revenue=revenue,
        )
        loan_summary = loan_schedule.summary if hasattr(loan_schedule, "summary") else pd.DataFrame()
        if isinstance(loan_summary, pd.DataFrame) and not loan_summary.empty:
            total_loan_draw = float(pd.to_numeric(loan_summary.get("Draw"), errors="coerce").fillna(0.0).sum())
        else:
            total_loan_draw = 0.0
        metrics.update(
            {
                "Corporate Tax Rate": tax_rate,
                "Investor Share": investor_share,
                "Owner Share": owner_share,
                "Terminal Growth Rate": _get_global("Terminal growth", 0.0),
                "Capital Gains Tax Rate": _get_global("Capital gains tax rate", 0.0),
                "Discount Rate": discount_rate,
                "Total Initial Investment": metrics.get("Initial Project Outlay", total_investment),
                "Initial Loan Funding": metrics.get("Initial Loan Draw", total_loan_draw),
                "Initial Equity Investment": metrics.get(
                    "Initial Equity Investment", total_investment - total_loan_draw
                ),
                "Scenario": scenario_name,
                "Planning Start Month": page.projection.planning_start,
            }
        )
        if not np.isnan(metrics.get("Payback Period (months)", float("nan"))):
            metrics["Payback Period (years)"] = metrics["Payback Period (months)"] / 12.0

        break_even = compute_break_even(revenue, cost_outputs)
        payback = compute_payback(
            financials,
            revenue,
            initial_project_outlay=metrics.get("Initial Project Outlay"),
        )

        results = {
            "depreciation": depreciation,
            "production": production,
            "revenue": revenue,
            "costs": cost_outputs,
            "loan_schedule": loan_schedule,
            "working_capital": working_capital,
            "financials": financials,
            "expenses": expenses,
            "metrics": metrics,
            "break_even": break_even,
            "payback": payback,
            "scenario": scenario_name,
            "input_page_snapshot": page,
            "staff_schedule": staff_schedule,
        }
        self._scenario_cache[scenario_name] = (signature, copy.deepcopy(results))
        return results

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        self._scenario_cache.clear()

    def input_signature(self) -> str:
        return self._input_signature()

    def result_signature(self, result: Dict[str, object]) -> str:
        return self._result_signature(result)

    def auto_build_all(
        self,
        scenarios: Iterable[str] | None = None,
        max_passes: int = 3,
    ) -> Dict[str, Dict[str, object]]:
        scenario_list = [s.upper() for s in (scenarios or self.SCENARIOS)]
        outputs: Dict[str, Dict[str, object]] = {}
        for scenario in scenario_list:
            previous = None
            last_result: Dict[str, object] | None = None
            for _ in range(max_passes):
                result = self.build(scenario)
                signature = self._result_signature(result)
                if previous is not None and signature == previous:
                    last_result = result
                    break
                previous = signature
                last_result = result
            if last_result is None:
                last_result = self.build(scenario)
            outputs[scenario] = last_result
        return outputs
