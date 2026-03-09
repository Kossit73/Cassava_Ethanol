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


    def _materialize_required_defaults(self, page: inputs.InputLandingPage) -> None:
        """Use seeded default tables when placeholders are still active."""

        required_tables = [
            page.global_inputs,
            page.initial_investment,
            page.revenue_inputs,
            page.production_monthly,
            page.loan_schedule,
        ]
        for table in required_tables:
            if table.placeholder and table.data is not None and not table.data.empty:
                table.set_data(table.data, mark_user_input=True)

    def _validate_required_inputs(self, page: inputs.InputLandingPage) -> None:
        """Hard validation gate for investor-grade completeness checks."""

        missing: list[str] = []
        required_tables = [
            ("Global Inputs", page.global_inputs.model_frame),
            ("Initial Investment", page.initial_investment.model_frame),
            ("Revenue Inputs", page.revenue_inputs.model_frame),
            ("Production Monthly", page.production_monthly.model_frame),
            ("Loan Schedule", page.loan_schedule.model_frame),
        ]
        for name, frame in required_tables:
            if frame is None or frame.empty:
                missing.append(name)

        if missing:
            raise ValueError("Missing required input tables: " + ", ".join(missing))

        globals_df = page.global_inputs.model_frame
        lookup = globals_df.set_index("Parameter")["Value"].to_dict() if not globals_df.empty else {}

        def _must(parameter: str) -> float:
            if parameter not in lookup:
                raise ValueError(f"Missing required global assumption: {parameter}")
            try:
                return float(lookup[parameter])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid numeric value for global assumption: {parameter}") from exc

        discount_rate = _must("Discount rate")
        tax_rate = _must("Corporate tax rate")
        if not (0.0 <= tax_rate <= 0.6):
            raise ValueError("Corporate tax rate must be between 0 and 0.60")
        if not (0.0 <= discount_rate <= 0.5):
            raise ValueError("Discount rate must be between 0 and 0.50")

        take_or_pay = _must("Take-or-pay share")
        if not (0.0 <= take_or_pay <= 1.0):
            raise ValueError("Take-or-pay share must be between 0 and 1")

        floor_price = _must("Offtake floor price (USD/L)")
        ceiling_price = _must("Offtake ceiling price (USD/L)")
        if ceiling_price < floor_price:
            raise ValueError("Offtake ceiling price must be greater than or equal to the offtake floor price")

        contracted_share = _must("Contracted feedstock share")
        if not (0.0 <= contracted_share <= 1.0):
            raise ValueError("Contracted feedstock share must be between 0 and 1")
        open_market_share = float(lookup.get("Open market feedstock share", 1.0 - contracted_share))
        if not (0.0 <= open_market_share <= 1.0):
            raise ValueError("Open market feedstock share must be between 0 and 1")
        if not np.isclose(contracted_share + open_market_share, 1.0, atol=1e-6):
            raise ValueError("Contracted feedstock share plus open market feedstock share must equal 1.0")

        # Projection horizon consistency for annual/monthly inputs.
        projection_start = pd.Period(f"{int(page.projection.start_year)}-01", freq="M")
        projection_end = pd.Period(f"{int(page.projection.end_year)}-12", freq="M")

        def _validate_year_column(df: pd.DataFrame, table_name: str, column: str = "Year") -> None:
            if df is None or df.empty or column not in df.columns:
                return
            years = pd.to_numeric(df[column], errors="coerce")
            if years.isna().any():
                raise ValueError(f"{table_name}: invalid year values detected")
            if ((years < page.projection.start_year) | (years > page.projection.end_year)).any():
                raise ValueError(
                    f"{table_name}: year values must be within projection horizon "
                    f"{page.projection.start_year}-{page.projection.end_year}"
                )

        def _validate_month_column(df: pd.DataFrame, table_name: str, column: str) -> None:
            if df is None or df.empty or column not in df.columns:
                return
            months = pd.to_datetime(df[column].astype(str), errors="coerce")
            if months.isna().any():
                raise ValueError(f"{table_name}: invalid month values detected in '{column}'")
            periods = months.dt.to_period("M")
            if ((periods < projection_start) | (periods > projection_end)).any():
                raise ValueError(
                    f"{table_name}: month values in '{column}' must fall within projection window "
                    f"{projection_start.strftime('%Y-%m')} to {projection_end.strftime('%Y-%m')}"
                )

        _validate_year_column(page.production_annual.model_frame, "Production Annual")
        _validate_year_column(page.inflation_schedule.model_frame, "Inflation Schedule")
        _validate_month_column(page.production_monthly.model_frame, "Production Monthly", "Start Month")
        _validate_month_column(page.direct_costs_monthly.model_frame, "Direct Costs Monthly", "Month")
        _validate_month_column(page.staff_costs_monthly.model_frame, "Staff Costs Monthly", "Month")
        _validate_month_column(page.other_opex_monthly.model_frame, "Other Opex Monthly", "Month")
        _validate_month_column(page.accounts_receivable.model_frame, "Accounts Receivable", "Effective Month")
        _validate_month_column(page.inventory_payable.model_frame, "Inventory/Payable", "Effective Month")
        _validate_month_column(page.loan_schedule.model_frame, "Loan Schedule", "Start Month")

        # Financing consistency checks.
        init_df = page.initial_investment.model_frame
        capex = float(pd.to_numeric(init_df.get("Cost"), errors="coerce").fillna(0.0).sum()) if not init_df.empty else 0.0
        loan_df = page.loan_schedule.model_frame
        debt_draw = float(
            pd.to_numeric(
                loan_df.get("Loan Amount", loan_df.get("Amount", loan_df.get("Draw Amount"))),
                errors="coerce",
            ).fillna(0.0).sum()
        ) if not loan_df.empty else 0.0
        if debt_draw - capex > 1e-6:
            raise ValueError("Total debt draw cannot exceed total initial investment envelope")

        investor_share = float(lookup.get("Investor share capital", 0.0))
        owner_share = float(lookup.get("Owner share capital", max(0.0, 1.0 - investor_share)))
        if not np.isclose(investor_share + owner_share, 1.0, atol=1e-6):
            raise ValueError("Investor share capital plus owner share capital must equal 1.0")
        implied_equity = capex - debt_draw
        if implied_equity < -1e-6:
            raise ValueError("Equity plus debt draw must reconcile to the initial capex envelope")

        # Revenue volume linkage: enforce only when sales-volume column exists.
        rev_df = page.revenue_inputs.model_frame
        prod_df = page.production_monthly.model_frame
        volume_cols = ["Volume", "Sales Volume", "Ethanol litres sold"]
        volume_col = next((c for c in volume_cols if c in rev_df.columns), None)
        if volume_col and not rev_df.empty and "Product" in rev_df.columns and not prod_df.empty:
            sold = pd.to_numeric(
                rev_df.loc[rev_df["Product"].astype(str).str.contains("ethanol", case=False, na=False), volume_col],
                errors="coerce",
            ).fillna(0.0).sum()
            produced = pd.to_numeric(prod_df.get("Ethanol litres"), errors="coerce").fillna(0.0).sum()
            inventory_draw_litres = 0.0
            inv_df = page.inventory_payable.model_frame
            if not inv_df.empty and "Metric" in inv_df.columns and "Value" in inv_df.columns:
                inventory_draw_litres = pd.to_numeric(
                    inv_df.loc[
                        inv_df["Metric"].astype(str).str.contains("inventory draw litres", case=False, na=False),
                        "Value",
                    ],
                    errors="coerce",
                ).fillna(0.0).sum()
            if sold > produced + inventory_draw_litres + 1e-6:
                raise ValueError("Revenue sales volume cannot exceed produced ethanol litres unless inventory draw is modeled")

    def _apply_debt_strategy_toggles(self, page: inputs.InputLandingPage) -> None:
        globals_df = page.global_inputs.model_frame
        if globals_df.empty:
            return
        lookup = globals_df.set_index("Parameter")["Value"].to_dict()

        def _get(name: str, default: float = 0.0) -> float:
            try:
                return float(lookup.get(name, default))
            except (TypeError, ValueError):
                return default

        sculpting = _get("Debt sculpting enabled", 0.0) >= 0.5
        target_dscr = _get("Target DSCR", 1.25)
        refinancing = _get("Refinancing enabled", 0.0) >= 0.5
        refinancing_year = int(_get("Refinancing year", page.projection.start_year + 3))
        refinancing_rate = _get("Refinancing interest rate", 0.0)

        loan_df = page.loan_schedule.model_frame
        if loan_df.empty:
            return

        adjusted = loan_df.copy()

        if sculpting:
            if "Grace Years" in adjusted.columns:
                grace = pd.to_numeric(adjusted["Grace Years"], errors="coerce").fillna(0.0)
                adjusted["Grace Years"] = np.maximum(grace, 2.0)
            if "Tenor Years" in adjusted.columns:
                tenor = pd.to_numeric(adjusted["Tenor Years"], errors="coerce").fillna(0.0)
                tenor_extension = 1.0 if target_dscr >= 1.2 else 2.0
                adjusted["Tenor Years"] = tenor + tenor_extension

        if "Tenor Years" in adjusted.columns:
            tenor = pd.to_numeric(adjusted["Tenor Years"], errors="coerce").fillna(8.0)
            adjusted["Tenor Years"] = np.clip(tenor, 3.0, 20.0)
        if "Grace Years" in adjusted.columns:
            grace = pd.to_numeric(adjusted["Grace Years"], errors="coerce").fillna(1.0)
            adjusted["Grace Years"] = np.clip(grace, 0.0, 5.0)
        if {"Tenor Years", "Grace Years"}.issubset(adjusted.columns):
            adjusted["Grace Years"] = np.minimum(
                pd.to_numeric(adjusted["Grace Years"], errors="coerce").fillna(0.0),
                pd.to_numeric(adjusted["Tenor Years"], errors="coerce").fillna(3.0) - 1.0,
            )

        if refinancing and refinancing_rate > 0 and "Interest Rate" in adjusted.columns:
            start_period = pd.Period(f"{refinancing_year}-01", freq="M")
            if "Start Month" in adjusted.columns:
                starts = pd.to_datetime(adjusted["Start Month"], errors="coerce")
                mask = starts.dt.to_period("M") <= start_period
            else:
                mask = pd.Series(True, index=adjusted.index)
            adjusted.loc[mask, "Interest Rate"] = refinancing_rate

        if not adjusted.equals(loan_df):
            page.loan_schedule.set_data(adjusted, mark_user_input=page.loan_schedule.placeholder)

    def _apply_risk_and_contract_mechanics(
        self,
        page: inputs.InputLandingPage,
        production,
        revenue,
        cost_outputs: Dict[str, object],
    ) -> Dict[str, float]:
        """Integrate risk register and commercial contract assumptions."""

        globals_df = page.global_inputs.model_frame
        lookup = globals_df.set_index("Parameter")["Value"].to_dict() if not globals_df.empty else {}

        def _get(name: str, default: float = 0.0) -> float:
            try:
                return float(lookup.get(name, default))
            except (TypeError, ValueError):
                return default

        floor_price = _get("Offtake floor price (USD/L)", 0.0)
        ceiling_price = _get("Offtake ceiling price (USD/L)", float("inf"))
        take_or_pay = float(np.clip(_get("Take-or-pay share", 1.0), 0.0, 1.0))
        contracted_share = float(np.clip(_get("Contracted feedstock share", 0.0), 0.0, 1.0))
        contracted_discount = float(np.clip(_get("Contract feedstock discount", 0.0), 0.0, 0.8))

        risk_df = page.risk_schedule.model_frame
        risk_score = 0.0
        if not risk_df.empty:
            impact_map = {"low": 0.35, "medium": 0.65, "high": 1.0}
            for _, row in risk_df.iterrows():
                try:
                    prob = float(row.get("Probability", 0.0))
                except (TypeError, ValueError):
                    prob = 0.0
                impact_raw = row.get("Impact", 0.0)
                if isinstance(impact_raw, str):
                    impact = impact_map.get(impact_raw.strip().lower(), 0.5)
                else:
                    try:
                        impact = float(impact_raw)
                    except (TypeError, ValueError):
                        impact = 0.5
                risk_score += max(0.0, prob) * max(0.0, impact)
        risk_intensity = float(np.clip(risk_score, 0.0, 1.0))

        monthly_rev = revenue.monthly.copy()
        if "Total Revenue" in monthly_rev.columns and not monthly_rev.empty:
            total_rev = pd.to_numeric(monthly_rev["Total Revenue"], errors="coerce").fillna(0.0)
            volume = pd.to_numeric(getattr(production, "monthly", pd.DataFrame()).get("Ethanol litres"), errors="coerce").fillna(0.0)
            implied_price = total_rev / volume.replace(0.0, np.nan)
            adjusted_price = implied_price.clip(lower=floor_price if floor_price > 0 else None, upper=ceiling_price)
            price_factor = (adjusted_price / implied_price.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
            top_factor = np.clip(take_or_pay + (1 - take_or_pay) * 0.7, 0.0, 1.0)
            risk_factor = max(0.0, 1.0 - 0.25 * risk_intensity)
            overall = price_factor * top_factor * risk_factor
            monthly_rev = monthly_rev.mul(overall, axis=0)
            monthly_rev["Total Revenue"] = pd.to_numeric(monthly_rev.sum(axis=1), errors="coerce").fillna(0.0)
            revenue.monthly = monthly_rev
            revenue.annual = monthly_rev.resample("YE").sum()
            revenue.annual.index = revenue.annual.index.year

        direct = cost_outputs.get("Direct Costs")
        feedstock_saving = contracted_share * contracted_discount
        risk_cost_uplift = 0.2 * risk_intensity
        cost_multiplier = max(0.0, 1.0 - feedstock_saving + risk_cost_uplift)
        if direct is not None and hasattr(direct, "monthly"):
            direct.monthly = direct.monthly * cost_multiplier
            direct.annual = direct.monthly.resample("YE").sum()
            direct.annual.index = direct.annual.index.year

        return {
            "Risk Score": risk_intensity,
            "Commercial Price Floor": floor_price,
            "Commercial Price Ceiling": ceiling_price,
            "Take-or-pay Share": take_or_pay,
            "Feedstock Contract Share": contracted_share,
            "Feedstock Contract Discount": contracted_discount,
            "Commercial Cost Multiplier": cost_multiplier,
        }

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
        self._materialize_required_defaults(page)
        self._validate_required_inputs(page)
        self._apply_debt_strategy_toggles(page)

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

        risk_commercial = self._apply_risk_and_contract_mechanics(page, production, revenue, cost_outputs)

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

        terminal_growth_rate = _get_global("Terminal growth", 0.0)
        capital_gains_tax_rate = _get_global("Capital gains tax rate", 0.0)

        metrics = compute_key_metrics(
            financials,
            discount_rate=discount_rate,
            investor_share=investor_share,
            owner_share=owner_share,
            revenue=revenue,
            terminal_growth_rate=terminal_growth_rate,
            capital_gains_tax_rate=capital_gains_tax_rate,
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
                "Terminal Growth Rate": terminal_growth_rate,
                "Capital Gains Tax Rate": capital_gains_tax_rate,
                "Discount Rate": discount_rate,
                "Total Initial Investment": metrics.get("Initial Project Outlay", total_investment),
                "Initial Loan Funding": metrics.get("Initial Loan Draw", total_loan_draw),
                "Initial Equity Investment": metrics.get(
                    "Initial Equity Investment", total_investment - total_loan_draw
                ),
                "Scenario": scenario_name,
                "Planning Start Month": page.projection.planning_start,
                **risk_commercial,
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
