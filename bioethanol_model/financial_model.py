from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple, TYPE_CHECKING

import hashlib
import numpy as np
import pandas as pd

from . import inputs
from .integrated_cycle import IntegratedCycleOutput, build_integrated_cycle
if TYPE_CHECKING:
    from .advanced_tools import AdvancedAnalyticsToolkit

from .schedules import (
    CostOutput,
    ProductionOutput,
    RevenueOutput,
    WorkingCapitalOutput,
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

    def _prepare_page_for_scenario(
        self,
        scenario: str,
        source_page: inputs.InputLandingPage | None = None,
    ) -> inputs.InputLandingPage:
        page = copy.deepcopy(source_page if source_page is not None else self.input_page)
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

        farm_share = float(
            np.clip(_get_global("Hybrid farm share", 0.0), 0.0, 1.0)
        )
        cycle_plan = page.annual_cycle_plan.model_frame
        if not cycle_plan.empty and "Hybrid Farm Share %" in cycle_plan.columns:
            cycle_shares = pd.to_numeric(
                cycle_plan["Hybrid Farm Share %"],
                errors="coerce",
            ).dropna()
            cycle_shares = cycle_shares.map(
                lambda value: value / 100.0 if abs(value) > 1.0 else value
            )
            if not cycle_shares.empty:
                farm_share = float(np.clip(cycle_shares.mean(), 0.0, 1.0))

        invest_df = page.initial_investment.model_frame
        if not invest_df.empty and "Item" in invest_df.columns:
            farm_assets = {
                str(value).strip().casefold()
                for value in page.farm_capex.model_frame.get("Asset", pd.Series(dtype=object))
                if str(value).strip()
            }
            item_names = invest_df["Item"].astype(str).str.strip().str.casefold()
            farm_mask = (
                item_names.isin(farm_assets)
                | invest_df["Item"].astype(str).str.contains("farm", case=False, na=False)
            )
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


    def _should_use_integrated_cycle(self, page: inputs.InputLandingPage) -> bool:
        """Return whether the cycle-ledger model is enabled for this input page."""

        frame = page.global_inputs.data
        enabled = 1.0
        if isinstance(frame, pd.DataFrame) and {"Parameter", "Value"}.issubset(frame.columns):
            mask = (
                frame["Parameter"].astype(str).str.strip().str.casefold()
                == "integrated cycle model enabled"
            )
            if mask.any():
                try:
                    enabled = float(frame.loc[mask, "Value"].iloc[-1])
                except (TypeError, ValueError):
                    enabled = 1.0
        if enabled < 0.5 or page.annual_cycle_plan.data.empty:
            return False

        # Backward compatibility: a saved legacy monthly schedule from an older
        # workbook should remain authoritative until the new cycle assumptions
        # are changed. Fresh defaults keep the monthly table as a placeholder,
        # so the integrated cycle engine remains the default for new projects.
        if not page.production_monthly.placeholder:
            default_cycle = inputs.default_input_page().annual_cycle_plan.data
            current_cycle = page.annual_cycle_plan.data.reset_index(drop=True)
            if current_cycle.equals(default_cycle.reset_index(drop=True)):
                return False
        return True

    def _materialize_required_defaults(
        self,
        page: inputs.InputLandingPage,
        *,
        integrated_cycle: bool = False,
    ) -> None:
        """Use seeded default tables when placeholders are still active."""

        required_tables = [
            page.global_inputs,
            page.initial_investment,
            page.revenue_inputs,
            page.loan_schedule,
        ]
        if integrated_cycle:
            required_tables.extend(
                [
                    page.annual_cycle_plan,
                    page.farm_cost_assumptions,
                    page.farm_capex,
                    page.procurement_plan,
                    page.product_routing,
                    page.commercialization_plan,
                    page.staff_positions,
                    page.staff_costs_monthly,
                    page.other_opex_monthly,
                ]
            )
        else:
            required_tables.append(page.production_monthly)
        for table in required_tables:
            if table.placeholder and table.data is not None and not table.data.empty:
                table.set_data(table.data, mark_user_input=True)

    def _sync_farm_capex_to_initial_investment(
        self,
        page: inputs.InputLandingPage,
    ) -> Dict[str, object]:
        """Keep FarmCo fixed assets inside the consolidated depreciation schedule."""

        farm = page.farm_capex.model_frame
        investment = page.initial_investment.model_frame
        if farm.empty or not {"Asset", "Cost"}.issubset(farm.columns):
            return {"applied": 0, "assets": []}
        if investment.empty:
            investment = pd.DataFrame(columns=page.initial_investment.columns)

        working = investment.copy()
        assets: list[str] = []
        for _, row in farm.iterrows():
            asset = str(row.get("Asset", "")).strip()
            if not asset:
                continue
            item_names = working.get("Item", pd.Series(dtype=object)).astype(str).str.strip().str.casefold()
            mask = item_names == asset.casefold()
            values = {
                "Item": asset,
                "Cost": max(0.0, float(pd.to_numeric(pd.Series([row.get("Cost")]), errors="coerce").fillna(0.0).iloc[0])),
                "Life (years)": max(0.0, float(pd.to_numeric(pd.Series([row.get("Life (years)")]), errors="coerce").fillna(0.0).iloc[0])),
                "Depreciation Rate": None,
                "Start Month": str(row.get("Start Month", page.projection.planning_start)),
            }
            if mask.any():
                for column, value in values.items():
                    if column in working.columns:
                        working.loc[mask, column] = value
            else:
                working = pd.concat([working, pd.DataFrame([values])], ignore_index=True)
            assets.append(asset)

        if not working.equals(investment):
            page.initial_investment.set_data(working, mark_user_input=True)
        return {"applied": len(assets), "assets": assets}

    def _normalize_global_units(self, page: inputs.InputLandingPage) -> Dict[str, object]:
        """Normalize percent-like global inputs so both 12 and 0.12 are accepted."""

        df = page.global_inputs.model_frame
        if df is None or df.empty:
            return {"passed": True, "converted": [], "outliers": [], "notes": ["No global assumptions supplied"]}

        if not {"Parameter", "Value"}.issubset(df.columns):
            return {"passed": False, "converted": [], "outliers": ["Global Inputs missing Parameter/Value columns"], "notes": []}

        working = df.copy()
        converted: list[str] = []
        outliers: list[str] = []
        explicit_percent_parameters = {
            "corporate tax rate",
            "investor share capital",
            "owner share capital",
            "terminal growth",
            "capital gains tax rate",
            "discount rate",
            "hybrid farm share",
            "take-or-pay share",
            "contracted feedstock share",
            "open market feedstock share",
            "contract feedstock discount",
            "refinancing interest rate",
            "repricing fee rate",
            "break cost rate",
            "raw cassava sorting reject %",
            "residue recovery to feed %",
            "farm transfer markup %",
            "default annual production increment %",
            "cash sweep share",
        }

        for idx, row in working.iterrows():
            parameter = str(row.get("Parameter", "")).strip()
            units = str(row.get("Units", "")).strip().lower()
            try:
                value = float(row.get("Value"))
            except (TypeError, ValueError):
                continue

            is_percent = ("%" in units) or (parameter.lower() in explicit_percent_parameters)
            if not is_percent:
                continue

            if 1.0 < value <= 100.0:
                working.at[idx, "Value"] = value / 100.0
                converted.append(parameter)
                value = value / 100.0

            if value < -1.0 or value > 2.0:
                outliers.append(f"{parameter}={value}")

        if not working.equals(df):
            page.global_inputs.set_data(working, mark_user_input=page.global_inputs.placeholder)

        return {
            "passed": len(outliers) == 0,
            "converted": converted,
            "outliers": outliers,
            "notes": ["Percent normalization: values in (1,100] are converted to decimal form by dividing by 100."],
        }

    def _auto_balance_global_inputs(self, page: inputs.InputLandingPage) -> Dict[str, object]:
        """Auto-balance linked global assumptions to reduce manual inconsistencies."""

        df = page.global_inputs.model_frame
        if df is None or df.empty or not {"Parameter", "Value"}.issubset(df.columns):
            return {"applied": 0, "details": []}

        working = df.copy()
        details: list[str] = []
        unit_map = working.set_index("Parameter")["Units"].to_dict() if "Units" in working.columns else {}

        def _read(parameter: str) -> float | None:
            mask = working["Parameter"].astype(str).str.strip() == parameter
            if not mask.any():
                return None
            raw = pd.to_numeric(working.loc[mask, "Value"], errors="coerce")
            if raw.empty:
                return None
            value = raw.iloc[0]
            if pd.isna(value):
                return None
            return float(value)

        def _write(parameter: str, value: float, reason: str) -> None:
            nonlocal working
            value = float(value)
            mask = working["Parameter"].astype(str).str.strip() == parameter
            if mask.any():
                current = pd.to_numeric(working.loc[mask, "Value"], errors="coerce").iloc[0]
                if pd.isna(current) or not np.isclose(float(current), value, atol=1e-9):
                    working.loc[mask, "Value"] = value
                    details.append(reason)
            else:
                row = {"Parameter": parameter, "Value": value}
                if "Units" in working.columns:
                    row["Units"] = unit_map.get(parameter, "")
                working = pd.concat([working, pd.DataFrame([row])], ignore_index=True)
                details.append(reason)
            return None

        investor_share = _read("Investor share capital")
        owner_share = _read("Owner share capital")
        if investor_share is not None or owner_share is not None:
            investor = max(0.0, investor_share if investor_share is not None else 0.0)
            owner = max(0.0, owner_share if owner_share is not None else 0.0)
            if investor_share is None and owner_share is not None:
                investor = max(0.0, 1.0 - owner)
            if owner_share is None and investor_share is not None:
                owner = max(0.0, 1.0 - investor)
            total = investor + owner
            if total > 0:
                investor /= total
                owner /= total
            _write("Investor share capital", investor, "Normalized investor/owner share capital split")
            _write("Owner share capital", owner, "Normalized investor/owner share capital split")

        contracted_share = _read("Contracted feedstock share")
        if contracted_share is not None:
            contracted = float(np.clip(contracted_share, 0.0, 1.0))
            _write("Contracted feedstock share", contracted, "Clamped contracted feedstock share to [0,1]")
            _write("Open market feedstock share", 1.0 - contracted, "Synced open market share to 1-contracted share")

        hybrid_share = _read("Hybrid farm share")
        if hybrid_share is not None:
            _write(
                "Hybrid farm share",
                float(np.clip(hybrid_share, 0.0, 1.0)),
                "Clamped hybrid farm share to [0,1]",
            )

        corporate_tax = _read("Corporate tax rate")
        if corporate_tax is not None:
            _write("Corporate tax rate", float(np.clip(corporate_tax, 0.0, 0.60)), "Clamped corporate tax rate to [0,0.60]")

        discount_rate = _read("Discount rate")
        if discount_rate is not None:
            _write("Discount rate", float(np.clip(discount_rate, 0.0, 0.50)), "Clamped discount rate to [0,0.50]")

        if not working.equals(df):
            page.global_inputs.set_data(working, mark_user_input=page.global_inputs.placeholder)

        return {"applied": len(details), "details": details}

    def _sync_tax_schedule_from_global_rate(self, page: inputs.InputLandingPage) -> Dict[str, object]:
        """Propagate global corporate tax assumption into the tax schedule table."""

        globals_df = page.global_inputs.model_frame
        tax_df = page.tax_schedule.model_frame
        if (
            globals_df is None
            or globals_df.empty
            or tax_df is None
            or tax_df.empty
            or not {"Parameter", "Value"}.issubset(globals_df.columns)
            or not {"Item", "Base Rate"}.issubset(tax_df.columns)
        ):
            return {"applied": False, "rows": 0}

        lookup = globals_df.set_index("Parameter")["Value"].to_dict()
        try:
            global_tax = float(lookup.get("Corporate tax rate"))
        except (TypeError, ValueError):
            return {"applied": False, "rows": 0}

        working = tax_df.copy()
        mask = working["Item"].astype(str).str.contains("corporate income tax", case=False, na=False)
        if not mask.any():
            return {"applied": False, "rows": 0}

        current = pd.to_numeric(working.loc[mask, "Base Rate"], errors="coerce")
        if current.empty or current.isna().all():
            working.loc[mask, "Base Rate"] = global_tax
            page.tax_schedule.set_data(working, mark_user_input=True)
            return {"applied": True, "rows": int(mask.sum())}

        return {"applied": False, "rows": int(mask.sum())}

    def _sync_production_annual_from_monthly(self, page: inputs.InputLandingPage) -> Dict[str, object]:
        """Derive production-annual input table from monthly inputs automatically."""

        monthly_df = page.production_monthly.model_frame
        if monthly_df is None or monthly_df.empty:
            return {"applied": False, "rows": 0}

        projection = page.projection
        production = compute_production_tables(
            monthly_df,
            projection.start_year,
            projection.end_year,
            planning_start=projection.planning_start_timestamp,
        )
        annual = getattr(production, "annual", pd.DataFrame())
        if not isinstance(annual, pd.DataFrame) or annual.empty:
            return {"applied": False, "rows": 0}

        annual_reset = annual.reset_index()
        if "index" in annual_reset.columns and "Year" not in annual_reset.columns:
            annual_reset = annual_reset.rename(columns={"index": "Year"})
        if "Year" not in annual_reset.columns:
            annual_reset["Year"] = annual_reset.index

        target_columns = page.production_annual.columns
        synced = pd.DataFrame(columns=target_columns)
        for column in target_columns:
            if column in annual_reset.columns:
                synced[column] = annual_reset[column]
            else:
                synced[column] = np.nan
        if "Year" in synced.columns:
            synced["Year"] = pd.to_numeric(synced["Year"], errors="coerce").fillna(0).astype(int)

        page.production_annual.set_data(synced, mark_user_input=True)
        return {"applied": True, "rows": int(len(synced))}

    @staticmethod
    def _loan_amount_column(loan_df: pd.DataFrame) -> str | None:
        for column in ("Loan Amount", "Amount", "Draw Amount", "Drawdown", "Draw"):
            if column in loan_df.columns:
                return column
        return None

    def _align_debt_to_capex_envelope(self, page: inputs.InputLandingPage) -> Dict[str, float]:
        """Keep scenario-specific debt draws within the available capex envelope."""

        init_df = page.initial_investment.model_frame
        capex = float(pd.to_numeric(init_df.get("Cost"), errors="coerce").fillna(0.0).sum()) if not init_df.empty else 0.0

        loan_df = page.loan_schedule.model_frame
        amount_column = self._loan_amount_column(loan_df)
        if loan_df.empty or amount_column is None:
            return {
                "Debt Funding Original Draw": 0.0,
                "Debt Funding Adjusted Draw": 0.0,
                "Debt Funding Reduction": 0.0,
            }

        loan_amounts = pd.to_numeric(loan_df[amount_column], errors="coerce").fillna(0.0)
        debt_draw = float(loan_amounts.sum())
        if debt_draw <= capex + 1e-6:
            return {
                "Debt Funding Original Draw": debt_draw,
                "Debt Funding Adjusted Draw": debt_draw,
                "Debt Funding Reduction": 0.0,
            }

        adjusted = loan_df.copy()
        scale = 0.0 if capex <= 0 else capex / debt_draw
        adjusted[amount_column] = loan_amounts * scale
        adjusted_draw = float(pd.to_numeric(adjusted[amount_column], errors="coerce").fillna(0.0).sum())
        page.loan_schedule.set_data(adjusted, mark_user_input=page.loan_schedule.placeholder)

        return {
            "Debt Funding Original Draw": debt_draw,
            "Debt Funding Adjusted Draw": adjusted_draw,
            "Debt Funding Reduction": max(0.0, debt_draw - adjusted_draw),
        }

    def _sanitize_loan_schedule_start_month(self, page: inputs.InputLandingPage) -> Dict[str, float]:
        """Normalize invalid loan start months to avoid hard-load crashes."""

        loan_df = page.loan_schedule.model_frame
        if loan_df is None or loan_df.empty or "Start Month" not in loan_df.columns:
            return {"Loan Start Month Fixes": 0.0}

        working = loan_df.copy()
        parsed = pd.to_datetime(working["Start Month"].astype(str), errors="coerce")
        invalid_mask = parsed.isna()
        if not invalid_mask.any():
            return {"Loan Start Month Fixes": 0.0}

        fallback_month = f"{int(page.projection.start_year):04d}-01"
        working.loc[invalid_mask, "Start Month"] = fallback_month
        page.loan_schedule.set_data(working, mark_user_input=True)
        return {"Loan Start Month Fixes": float(int(invalid_mask.sum()))}

    def _validate_required_inputs(
        self,
        page: inputs.InputLandingPage,
        *,
        integrated_cycle: bool = False,
    ) -> None:
        """Hard validation gate for investor-grade completeness checks."""

        missing: list[str] = []
        required_tables = [
            ("Global Inputs", page.global_inputs.model_frame),
            ("Revenue Inputs", page.revenue_inputs.model_frame),
            ("Initial Investment", page.initial_investment.model_frame),
            ("Loan Schedule", page.loan_schedule.model_frame),
        ]
        if integrated_cycle:
            required_tables.extend(
                [
                    ("Annual Cycle Plan", page.annual_cycle_plan.model_frame),
                    ("Farm Cost Assumptions", page.farm_cost_assumptions.model_frame),
                    ("Farm Capex", page.farm_capex.model_frame),
                    ("Procurement Plan", page.procurement_plan.model_frame),
                    ("Product Routing", page.product_routing.model_frame),
                    ("Commercialization Plan", page.commercialization_plan.model_frame),
                ]
            )
        else:
            required_tables.extend(
                [
                    ("Production Monthly", page.production_monthly.model_frame),
                ]
            )

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

        # Projection horizon consistency for annual/monthly inputs.
        projection_start = pd.Period(f"{int(page.projection.start_year)}-01", freq="M")
        projection_end = pd.Period(f"{int(page.projection.end_year)}-12", freq="M")

        def _validate_year_column(
            df: pd.DataFrame,
            table_name: str,
            column: str = "Year",
            *,
            enforce_projection_start: bool = True,
        ) -> None:
            if df is None or df.empty or column not in df.columns:
                return
            years = pd.to_numeric(df[column], errors="coerce")
            if years.isna().any():
                raise ValueError(f"{table_name}: invalid year values detected")
            if (years > page.projection.end_year).any():
                raise ValueError(
                    f"{table_name}: year values must be within projection horizon "
                    f"{page.projection.start_year}-{page.projection.end_year}"
                )
            if enforce_projection_start and (years < page.projection.start_year).any():
                raise ValueError(
                    f"{table_name}: year values must be within projection horizon "
                    f"{page.projection.start_year}-{page.projection.end_year}"
                )

        def _validate_month_column(
            df: pd.DataFrame,
            table_name: str,
            column: str,
            *,
            enforce_projection_start: bool = True,
        ) -> None:
            if df is None or df.empty or column not in df.columns:
                return
            months = pd.to_datetime(df[column].astype(str), errors="coerce")
            if months.isna().any():
                raise ValueError(f"{table_name}: invalid month values detected in '{column}'")
            periods = months.dt.to_period("M")
            if (periods > projection_end).any():
                raise ValueError(
                    f"{table_name}: month values in '{column}' must fall within projection window "
                    f"{projection_start.strftime('%Y-%m')} to {projection_end.strftime('%Y-%m')}"
                )
            if enforce_projection_start and (periods < projection_start).any():
                raise ValueError(
                    f"{table_name}: month values in '{column}' must fall within projection window "
                    f"{projection_start.strftime('%Y-%m')} to {projection_end.strftime('%Y-%m')}"
                )

        if integrated_cycle:
            cycle_frame = page.annual_cycle_plan.model_frame
            _validate_year_column(cycle_frame, "Annual Cycle Plan")
            for column in ("Planting Month", "Harvest Month", "Processing Start Month"):
                _validate_month_column(cycle_frame, "Annual Cycle Plan", column)
            cultivation = pd.to_numeric(cycle_frame.get("Cultivation Months"), errors="coerce")
            if cultivation.isna().any() or not cultivation.between(9, 12).all():
                raise ValueError("Annual Cycle Plan: Cultivation Months must be between 9 and 12")
            processing = pd.to_numeric(cycle_frame.get("Processing Months"), errors="coerce")
            if processing.isna().any() or not (processing == 3).all():
                raise ValueError("Annual Cycle Plan: Processing Months must equal 3")

            routing = page.product_routing.model_frame.copy()
            required_products = {
                "fuel ethanol",
                "hqcf",
                "garri",
                "industrial starch",
                "dextrin",
                "glucose syrup",
                "sorbitol",
                "animal feed",
            }
            supplied = {
                str(value).strip().casefold()
                for value in routing.get("Output Stream", pd.Series(dtype=object))
            }
            missing_products = sorted(required_products - supplied)
            if missing_products:
                raise ValueError(
                    "Product Routing: missing required saleable routes: "
                    + ", ".join(missing_products)
                )

        _validate_year_column(page.production_annual.model_frame, "Production Annual")
        _validate_year_column(
            page.inflation_schedule.model_frame,
            "Inflation Schedule",
            enforce_projection_start=False,
        )
        if not integrated_cycle:
            _validate_month_column(page.production_monthly.model_frame, "Production Monthly", "Start Month")
        _validate_month_column(page.direct_costs_monthly.model_frame, "Direct Costs Monthly", "Month")
        _validate_month_column(page.staff_costs_monthly.model_frame, "Staff Costs Monthly", "Month")
        _validate_month_column(page.other_opex_monthly.model_frame, "Other Opex Monthly", "Month")
        _validate_month_column(page.accounts_receivable.model_frame, "Accounts Receivable", "Effective Month")
        _validate_month_column(page.inventory_payable.model_frame, "Inventory/Payable", "Effective Month")
        _validate_month_column(
            page.loan_schedule.model_frame,
            "Loan Schedule",
            "Start Month",
            enforce_projection_start=False,
        )

        # Financing consistency checks.
        init_df = page.initial_investment.model_frame
        capex = float(pd.to_numeric(init_df.get("Cost"), errors="coerce").fillna(0.0).sum()) if not init_df.empty else 0.0
        loan_df = page.loan_schedule.model_frame
        amount_column = self._loan_amount_column(loan_df)
        debt_draw = (
            float(pd.to_numeric(loan_df[amount_column], errors="coerce").fillna(0.0).sum())
            if not loan_df.empty and amount_column is not None
            else 0.0
        )
        if debt_draw - capex > 1e-6:
            raise ValueError("Total debt draw cannot exceed total initial investment envelope")

        investor_share = float(lookup.get("Investor share capital", 0.0))
        owner_share = float(lookup.get("Owner share capital", max(0.0, 1.0 - investor_share)))
        if investor_share > 1.0 or owner_share > 1.0:
            if 0.0 <= investor_share <= 100.0 and 0.0 <= owner_share <= 100.0:
                investor_share /= 100.0
                owner_share /= 100.0
            else:
                raise ValueError("Investor/Owner share capital must be expressed as decimal fractions or percentages")
        total_share = investor_share + owner_share
        if total_share <= 0:
            raise ValueError("Investor share capital plus owner share capital must be positive")
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
        loan_schedule,
        *,
        integrated_cycle: bool = False,
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
        stress_vectors = {
            "volume": 0.0,
            "price": 0.0,
            "cost": 0.0,
            "schedule": 0.0,
        }
        p90_vectors = {
            "volume": 0.0,
            "price": 0.0,
            "cost": 0.0,
            "schedule": 0.0,
        }
        duration_vectors = {
            "volume": 0.0,
            "price": 0.0,
            "cost": 0.0,
            "schedule": 0.0,
        }
        risk_score = 0.0
        if not risk_df.empty:
            impact_map = {"low": 0.35, "medium": 0.65, "high": 1.0}
            class_map = {
                "volume": "volume",
                "yield": "volume",
                "logistics": "volume",
                "price": "price",
                "market": "price",
                "cost": "cost",
                "feedstock": "cost",
                "energy": "cost",
                "schedule": "schedule",
                "construction": "schedule",
                "delay": "schedule",
            }
            for _, row in risk_df.iterrows():
                try:
                    prob = float(row.get("Probability", 0.0))
                except (TypeError, ValueError):
                    prob = 0.0
                prob = float(np.clip(prob, 0.0, 1.0))

                impact_raw = row.get("Impact", 0.0)
                if isinstance(impact_raw, str):
                    impact = impact_map.get(impact_raw.strip().lower(), 0.5)
                else:
                    try:
                        impact = float(impact_raw)
                    except (TypeError, ValueError):
                        impact = 0.5
                impact = max(0.0, impact)

                expected_raw = row.get("Expected Impact", impact)
                try:
                    expected_impact = float(expected_raw)
                except (TypeError, ValueError):
                    expected_impact = impact
                expected_impact = max(0.0, expected_impact)

                downside_raw = row.get("P90 Downside", expected_impact)
                try:
                    p90_downside = float(downside_raw)
                except (TypeError, ValueError):
                    p90_downside = expected_impact
                p90_downside = max(expected_impact, p90_downside)

                duration_raw = row.get("Duration Months", 0.0)
                try:
                    duration_months = max(0.0, float(duration_raw))
                except (TypeError, ValueError):
                    duration_months = 0.0

                class_raw = str(row.get("Class", "")).strip().lower()
                risk_name = str(row.get("Risk", "")).strip().lower()
                key = class_map.get(class_raw)
                if key is None:
                    key = next((v for k, v in class_map.items() if k in risk_name), "cost")

                expected_component = prob * expected_impact
                downside_component = prob * p90_downside
                stress_vectors[key] += expected_component
                p90_vectors[key] += downside_component
                duration_vectors[key] += prob * duration_months

                risk_score += prob * impact

        for k in list(stress_vectors.keys()):
            stress_vectors[k] = float(np.clip(stress_vectors[k], 0.0, 1.0))
            p90_vectors[k] = float(np.clip(p90_vectors[k], 0.0, 1.5))
            duration_vectors[k] = float(max(0.0, duration_vectors[k]))

        risk_intensity = float(np.clip(risk_score, 0.0, 1.0))
        volume_stress = stress_vectors["volume"]
        price_stress = stress_vectors["price"]
        cost_stress = stress_vectors["cost"]
        schedule_stress = stress_vectors["schedule"]

        monthly_rev = revenue.monthly.copy()
        if "Total Revenue" in monthly_rev.columns and not monthly_rev.empty:
            total_rev = pd.to_numeric(monthly_rev["Total Revenue"], errors="coerce").fillna(0.0)
            volume = pd.to_numeric(
                getattr(production, "monthly", pd.DataFrame()).get("Ethanol litres"),
                errors="coerce",
            ).fillna(0.0)
            if integrated_cycle:
                price_factor = pd.Series(1.0, index=monthly_rev.index)
                top_factor = 1.0
            else:
                implied_price = total_rev / volume.replace(0.0, np.nan)
                adjusted_price = implied_price.clip(
                    lower=floor_price if floor_price > 0 else None,
                    upper=ceiling_price,
                )
                price_factor = (
                    adjusted_price / implied_price.replace(0.0, np.nan)
                ).replace([np.inf, -np.inf], np.nan).fillna(1.0)
                top_factor = np.clip(
                    take_or_pay + (1 - take_or_pay) * 0.7,
                    0.0,
                    1.0,
                )
            price_risk_factor = max(0.0, 1.0 - price_stress)
            volume_risk_factor = max(0.0, 1.0 - volume_stress)
            schedule_risk_factor = max(0.0, 1.0 - 0.5 * schedule_stress)
            overall = (
                price_factor
                * top_factor
                * price_risk_factor
                * volume_risk_factor
                * schedule_risk_factor
            )
            revenue_columns = [
                column for column in monthly_rev.columns if column != "Total Revenue"
            ]
            if integrated_cycle:
                monthly_rev.loc[:, revenue_columns] = monthly_rev[revenue_columns].mul(
                    overall,
                    axis=0,
                )
                monthly_rev["Total Revenue"] = monthly_rev[revenue_columns].sum(axis=1)
            else:
                # Preserve the historical legacy-mode contract for previously
                # saved workbooks. Integrated mode uses the corrected product-only
                # roll-up above.
                monthly_rev = monthly_rev.mul(overall, axis=0)
                monthly_rev["Total Revenue"] = monthly_rev.sum(axis=1)

            schedule_delay_months = int(round(duration_vectors["schedule"] * schedule_stress))
            if schedule_delay_months > 0:
                monthly_rev = monthly_rev.shift(schedule_delay_months, fill_value=0.0)

            revenue.monthly = monthly_rev
            revenue.annual = monthly_rev.groupby(monthly_rev.index.year).sum()
            revenue.annual.index.name = "Year"

            monthly_prod = getattr(production, "monthly", pd.DataFrame())
            if isinstance(monthly_prod, pd.DataFrame) and not monthly_prod.empty:
                monthly_prod_adj = monthly_prod.copy()
                if "Ethanol litres" in monthly_prod_adj.columns:
                    monthly_prod_adj["Ethanol litres"] = pd.to_numeric(monthly_prod_adj["Ethanol litres"], errors="coerce").fillna(0.0) * volume_risk_factor
                if "Cassava ton" in monthly_prod_adj.columns:
                    monthly_prod_adj["Cassava ton"] = pd.to_numeric(monthly_prod_adj["Cassava ton"], errors="coerce").fillna(0.0) * volume_risk_factor
                if schedule_delay_months > 0:
                    monthly_prod_adj = monthly_prod_adj.shift(schedule_delay_months, fill_value=0.0)
                production.monthly = monthly_prod_adj
                production.annual = monthly_prod_adj.groupby(monthly_prod_adj.index.year).sum()
                production.annual.index.name = "Year"

        direct = cost_outputs.get("Direct Costs")
        feedstock_saving = (
            0.0 if integrated_cycle else contracted_share * contracted_discount
        )
        risk_cost_uplift = 0.2 * risk_intensity + 0.35 * cost_stress
        cost_multiplier = max(0.0, 1.0 - feedstock_saving + risk_cost_uplift)
        if direct is not None and hasattr(direct, "monthly"):
            direct.monthly = direct.monthly * cost_multiplier
            direct.annual = direct.monthly.groupby(direct.monthly.index.year).sum()
            direct.annual.index.name = "Year"

        schedule_delay_months = int(round(duration_vectors["schedule"] * schedule_stress))
        if schedule_delay_months > 0 and loan_schedule is not None and hasattr(loan_schedule, "schedule"):
            debt_schedule = getattr(loan_schedule, "schedule", pd.DataFrame())
            if isinstance(debt_schedule, pd.DataFrame) and not debt_schedule.empty and "Month" in debt_schedule.columns:
                shifted = debt_schedule.copy()
                shifted["Month"] = pd.to_datetime(shifted["Month"], errors="coerce") + pd.DateOffset(months=schedule_delay_months)
                loan_schedule.schedule = shifted
                if hasattr(loan_schedule, "annual") and isinstance(loan_schedule.annual, pd.DataFrame) and not loan_schedule.annual.empty:
                    annual = shifted.copy()
                    annual["Year"] = pd.to_datetime(annual["Month"], errors="coerce").dt.year
                    loan_schedule.annual = (
                        annual.groupby(["Loan", "Year"]).agg(
                            Interest_Rate=("Interest Rate", "first"),
                            Yearly_Remaining_Balance=("Opening Balance", "first"),
                            Interest_Paid=("Interest", "sum"),
                            Principal_Paid=("Principal", "sum"),
                            Total_Payment=("Payment", "sum"),
                            Year_End_Balance=("Closing Balance", "last"),
                        )
                        .reset_index()
                        .rename(
                            columns={
                                "Interest_Rate": "Interest Rate",
                                "Yearly_Remaining_Balance": "Yearly Remaining Balance",
                                "Interest_Paid": "Interest Paid",
                                "Principal_Paid": "Principal Paid",
                                "Total_Payment": "Total Payment",
                                "Year_End_Balance": "Year-End Balance",
                            }
                        )
                    )

        return {
            "Risk Score": risk_intensity,
            "Commercial Price Floor": floor_price,
            "Commercial Price Ceiling": ceiling_price,
            "Take-or-pay Share": take_or_pay,
            "Feedstock Contract Share": contracted_share,
            "Feedstock Contract Discount": contracted_discount,
            "Commercial Cost Multiplier": cost_multiplier,
            "Risk Volume Stress (EV)": volume_stress,
            "Risk Price Stress (EV)": price_stress,
            "Risk Cost Stress (EV)": cost_stress,
            "Risk Schedule Stress (EV)": schedule_stress,
            "Risk Volume Stress (P90)": p90_vectors["volume"],
            "Risk Price Stress (P90)": p90_vectors["price"],
            "Risk Cost Stress (P90)": p90_vectors["cost"],
            "Risk Schedule Stress (P90)": p90_vectors["schedule"],
            "Risk Volume Duration (months, EV)": duration_vectors["volume"],
            "Risk Price Duration (months, EV)": duration_vectors["price"],
            "Risk Cost Duration (months, EV)": duration_vectors["cost"],
            "Risk Schedule Duration (months, EV)": duration_vectors["schedule"],
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

    def _sync_cassava_feedstock_costs(
        self,
        page: inputs.InputLandingPage,
        production,
        scenario: str,
    ) -> Dict[str, float]:
        """Derive monthly cassava feedstock costs from tonnage and price-per-ton."""

        globals_df = page.global_inputs.model_frame
        lookup = globals_df.set_index("Parameter")["Value"].to_dict() if not globals_df.empty else {}

        def _get(name: str, default: float = 0.0) -> float:
            try:
                return float(lookup.get(name, default))
            except (TypeError, ValueError):
                return default

        farm_cost = max(0.0, _get("Cassava farm cost per ton", 0.0))
        purchase_cost = max(0.0, _get("Cassava purchase cost per ton", 0.0))
        farm_share = float(np.clip(_get("Hybrid farm share", 0.0), 0.0, 1.0))

        scenario_name = str(scenario).upper()
        if scenario_name == "BUY_ONLY":
            effective_cost_per_ton = purchase_cost
        elif scenario_name == "HYBRID":
            effective_cost_per_ton = (farm_share * farm_cost) + ((1.0 - farm_share) * purchase_cost)
        else:
            effective_cost_per_ton = farm_cost

        monthly_prod = getattr(production, "monthly", pd.DataFrame())
        if not isinstance(monthly_prod, pd.DataFrame) or monthly_prod.empty or "Cassava ton" not in monthly_prod.columns:
            return {"Cassava Feedstock Cost Per Ton (effective)": effective_cost_per_ton}

        cassava_ton = pd.to_numeric(monthly_prod["Cassava ton"], errors="coerce").fillna(0.0)
        if cassava_ton.empty:
            return {"Cassava Feedstock Cost Per Ton (effective)": effective_cost_per_ton}

        feedstock_amount = (cassava_ton * effective_cost_per_ton).astype(float)
        feedstock_rows = pd.DataFrame(
            {
                "Month": pd.to_datetime(feedstock_amount.index, errors="coerce").strftime("%Y-%m"),
                "Cost Category": "Cassava Feedstock",
                "Amount": feedstock_amount.values,
            }
        )
        feedstock_rows = feedstock_rows.dropna(subset=["Month"])

        direct_df = page.direct_costs_monthly.model_frame
        if direct_df is None or direct_df.empty:
            non_feedstock = pd.DataFrame(columns=["Month", "Cost Category", "Amount"])
        else:
            normalized = direct_df.copy()
            if not {"Month", "Cost Category", "Amount"}.issubset(normalized.columns):
                normalized = pd.DataFrame(columns=["Month", "Cost Category", "Amount"])
            else:
                mask = normalized["Cost Category"].astype(str).str.strip().str.casefold() == "cassava feedstock"
                non_feedstock = normalized.loc[~mask, ["Month", "Cost Category", "Amount"]]

        if non_feedstock.empty:
            updated_direct = feedstock_rows.copy()
        else:
            updated_direct = pd.concat([non_feedstock, feedstock_rows], ignore_index=True)
        page.direct_costs_monthly.set_data(updated_direct, mark_user_input=True)

        return {"Cassava Feedstock Cost Per Ton (effective)": effective_cost_per_ton}

    def _apply_dynamic_debt_mechanics(
        self,
        page: inputs.InputLandingPage,
        loan_schedule,
        financials,
    ) -> Dict[str, float]:
        """Apply DSCR-based sculpting, refinancing economics, and covenant package."""

        globals_df = page.global_inputs.model_frame
        lookup = globals_df.set_index("Parameter")["Value"].to_dict() if not globals_df.empty else {}

        def _get(name: str, default: float = 0.0) -> float:
            try:
                return float(lookup.get(name, default))
            except (TypeError, ValueError):
                return default

        sculpting = _get("Debt sculpting enabled", 0.0) >= 0.5
        target_dscr = max(0.5, _get("Target DSCR", 1.25))
        refinancing = _get("Refinancing enabled", 0.0) >= 0.5
        refinancing_year = int(_get("Refinancing year", page.projection.start_year + 3))
        repricing_fee_rate = max(0.0, _get("Repricing fee rate", 0.0))
        break_cost_rate = max(0.0, _get("Break cost rate", 0.0))
        dsra_months = max(0.0, _get("DSRA months", 0.0))
        lockup_threshold = max(0.0, _get("DSCR lock-up threshold", 1.0))
        cash_sweep_trigger = max(lockup_threshold, _get("Cash sweep trigger DSCR", 1.35))
        cash_sweep_share = float(np.clip(_get("Cash sweep share", 0.0), 0.0, 1.0))
        cure_window = max(1, int(round(_get("Breach cure window months", 3.0))))

        schedule = getattr(loan_schedule, "schedule", pd.DataFrame())
        if not isinstance(schedule, pd.DataFrame) or schedule.empty:
            return {}

        cfads = pd.to_numeric(financials.cashflow_monthly.get("Operating Cash Flow"), errors="coerce").fillna(0.0)
        cfads_by_month = cfads.to_dict()

        refined = schedule.copy().sort_values(["Loan", "Month"]).reset_index(drop=True)
        refined["Month"] = pd.to_datetime(refined["Month"], errors="coerce")
        refinance_date = pd.Timestamp(refinancing_year, 1, 1)
        refinancing_cost_total = 0.0
        cash_sweep_total = 0.0
        lockup_months = 0
        cured_breaches = 0

        for loan_name, idx in refined.groupby("Loan", sort=False).groups.items():
            opening = 0.0
            loan_idx = list(idx)
            refinanced_once = False
            breach_flags: list[bool] = []

            for pos, i in enumerate(loan_idx):
                row = refined.loc[i]
                draw = float(pd.to_numeric(row.get("Draw"), errors="coerce") or 0.0)
                rate = float(pd.to_numeric(row.get("Interest Rate"), errors="coerce") or 0.0)
                month = pd.to_datetime(row.get("Month"), errors="coerce")

                opening = max(0.0, opening + draw)
                interest = opening * max(0.0, rate / 12.0)
                original_principal = float(pd.to_numeric(row.get("Principal"), errors="coerce") or 0.0)

                principal = original_principal
                if sculpting and opening > 0 and original_principal > 0:
                    cfads_m = float(cfads_by_month.get(month, 0.0))
                    target_service = max(0.0, cfads_m / target_dscr)
                    principal = np.clip(target_service - interest, 0.0, opening)

                    if target_service > 0:
                        dscr_m = cfads_m / max(interest + principal, 1e-9)
                    else:
                        dscr_m = float("inf") if cfads_m > 0 else 0.0
                    if dscr_m < lockup_threshold:
                        lockup_months += 1
                    breach_flags.append(dscr_m < 1.0)

                    if dscr_m > cash_sweep_trigger and opening > principal:
                        excess_cfads = max(0.0, cfads_m - (cash_sweep_trigger * (interest + principal)))
                        sweep = min(opening - principal, excess_cfads * cash_sweep_share)
                        principal += sweep
                        cash_sweep_total += sweep

                closing = max(0.0, opening - principal)
                payment = interest + principal

                if refinancing and (not refinanced_once) and month >= refinance_date and opening > 0:
                    one_off_cost = opening * (repricing_fee_rate + break_cost_rate)
                    interest += one_off_cost
                    payment += one_off_cost
                    refinancing_cost_total += one_off_cost
                    refinanced_once = True

                refined.loc[i, "Opening Balance"] = opening
                refined.loc[i, "Interest"] = interest
                refined.loc[i, "Principal"] = principal
                refined.loc[i, "Payment"] = payment
                refined.loc[i, "Closing Balance"] = closing
                opening = closing

            for j, flag in enumerate(breach_flags):
                if not flag:
                    continue
                window = breach_flags[j + 1 : j + 1 + cure_window]
                if window and not any(window):
                    cured_breaches += 1

        loan_schedule.schedule = refined
        if not refined.empty:
            loan_schedule.summary = refined.groupby("Loan").agg({"Draw": "sum", "Interest": "sum", "Principal": "sum", "Payment": "sum"})
            annual = (
                refined.assign(Year=refined["Month"].dt.year)
                .groupby(["Loan", "Year"]).agg(
                    Interest_Rate=("Interest Rate", "first"),
                    Yearly_Remaining_Balance=("Opening Balance", "first"),
                    Interest_Paid=("Interest", "sum"),
                    Principal_Paid=("Principal", "sum"),
                    Total_Payment=("Payment", "sum"),
                    Year_End_Balance=("Closing Balance", "last"),
                )
                .reset_index()
                .rename(
                    columns={
                        "Interest_Rate": "Interest Rate",
                        "Yearly_Remaining_Balance": "Yearly Remaining Balance",
                        "Interest_Paid": "Interest Paid",
                        "Principal_Paid": "Principal Paid",
                        "Total_Payment": "Total Payment",
                        "Year_End_Balance": "Year-End Balance",
                    }
                )
            )
            annual["Monthly Interest (Balance × Rate / 12)"] = annual["Yearly Remaining Balance"] * annual["Interest Rate"] / 12.0
            loan_schedule.annual = annual

        avg_debt_service = float(pd.to_numeric(refined.get("Payment"), errors="coerce").fillna(0.0).mean()) if not refined.empty else 0.0
        dsra_reset_amount = avg_debt_service * dsra_months

        return {
            "Refinancing Economics Cost": refinancing_cost_total,
            "Cash Sweep Applied": cash_sweep_total,
            "DSRA Reset Amount": dsra_reset_amount,
            "DSCR Lock-up Months": float(lockup_months),
            "Breach Cure Assumed Months": float(cured_breaches),
            "Cash Sweep Trigger DSCR": cash_sweep_trigger,
            "Cash Sweep Share": cash_sweep_share,
        }

    def _compute_bankability_scorecard(self, metrics: Dict[str, float], assumption_audit: Dict[str, object] | None = None) -> Dict[str, float]:
        """Compute concise bankability scorecard dimensions (0-100)."""

        def _num(key: str, default: float = float("nan")) -> float:
            try:
                return float(metrics.get(key, default))
            except (TypeError, ValueError):
                return default

        def _score_linear(value: float, low: float, high: float) -> float:
            if not np.isfinite(value):
                return 0.0
            if high <= low:
                return 0.0
            return float(np.clip((value - low) / (high - low), 0.0, 1.0) * 100.0)

        # Returns quality: combine NPV sign/magnitude and IRR level.
        npv = _num("Project NPV", 0.0)
        irr = _num("Project IRR", float("nan"))
        npv_component = 100.0 if npv > 0 else 0.0
        irr_component = _score_linear(irr, 0.10, 0.25)
        returns_quality = 0.5 * npv_component + 0.5 * irr_component

        # Debt capacity: DSCR/LLCR/PLCR jointly.
        dscr_min = _num("DSCR (min)")
        llcr = _num("LLCR")
        plcr = _num("PLCR")
        debt_capacity = float(np.mean([
            _score_linear(dscr_min, 1.0, 1.5),
            _score_linear(llcr, 1.1, 1.8),
            _score_linear(plcr, 1.1, 1.8),
        ]))

        # Liquidity resilience: minimum cash balance and liquidity cover months.
        min_cash = _num("Minimum Monthly Cash Balance", 0.0)
        liq_cover = _num("Months of Liquidity Cover (min)", _num("Liquidity Cover (months)", 0.0))
        liquidity_resilience = float(np.mean([
            100.0 if min_cash >= 0 else 0.0,
            _score_linear(liq_cover, 1.5, 6.0),
        ]))

        # Contract coverage quality: take-or-pay and contracted feedstock assumptions.
        take_or_pay = _num("Take-or-pay share", _num("Take-or-pay Share", 0.0))
        contracted_share = _num("Contracted feedstock share", _num("Contracted Feedstock Share", 0.0))
        contract_coverage = float(np.mean([
            _score_linear(take_or_pay, 0.5, 0.9),
            _score_linear(contracted_share, 0.4, 0.8),
        ]))

        # Risk concentration: lower risk score is better.
        risk_score = _num("Risk Score", 0.0)
        risk_concentration = float(np.clip((1.0 - np.clip(risk_score, 0.0, 1.0)) * 100.0, 0.0, 100.0))

        # Governance/completeness checks from assumption/invariant gates.
        quality_pass = bool((assumption_audit or {}).get("passed", False))
        inv_bal = _num("Invariant Balance Sheet Balanced", 0.0)
        inv_cf = _num("Invariant Cash Flow Bridge Consistent", 0.0)
        inv_debt = _num("Invariant Debt Rollforward Consistent", 0.0)
        governance = float(np.mean([
            100.0 if quality_pass else 0.0,
            100.0 if inv_bal >= 1.0 else 0.0,
            100.0 if inv_cf >= 1.0 else 0.0,
            100.0 if inv_debt >= 1.0 else 0.0,
        ]))

        dimensions = {
            "Bankability Returns Quality Score": returns_quality,
            "Bankability Debt Capacity Score": debt_capacity,
            "Bankability Liquidity Resilience Score": liquidity_resilience,
            "Bankability Contract Coverage Quality Score": contract_coverage,
            "Bankability Risk Concentration Score": risk_concentration,
            "Bankability Governance Completeness Score": governance,
        }
        overall = float(np.mean(list(dimensions.values()))) if dimensions else float("nan")
        dimensions["Bankability Scorecard Overall"] = overall
        return dimensions

    def _compute_accounting_invariants(self, financials, loan_schedule) -> Dict[str, float]:
        """Return invariant checks used for model-drift detection and audit."""

        out: Dict[str, float] = {}

        # 1) Annual balance-sheet balancing.
        bal_annual = getattr(financials, "balance_annual", pd.DataFrame())
        if isinstance(bal_annual, pd.DataFrame) and not bal_annual.empty:
            assets = pd.to_numeric(bal_annual.get("Total Assets"), errors="coerce")
            liab_eq = pd.to_numeric(bal_annual.get("Total Liabilities & Equity"), errors="coerce")
            delta = (assets - liab_eq).abs()
            out["Invariant Balance Sheet Max Delta"] = float(delta.max()) if not delta.dropna().empty else float("nan")
            out["Invariant Balance Sheet Balanced"] = float(1.0 if (delta.fillna(0.0) < 1e-3).all() else 0.0)
        else:
            out["Invariant Balance Sheet Max Delta"] = float("nan")
            out["Invariant Balance Sheet Balanced"] = float("nan")

        # 2) Cash-flow bridge consistency.
        cf = getattr(financials, "cashflow_monthly", pd.DataFrame())
        if isinstance(cf, pd.DataFrame) and not cf.empty:
            operating = pd.to_numeric(cf.get("Operating Cash Flow"), errors="coerce").fillna(0.0)
            investing = pd.to_numeric(cf.get("Investing Cash Flow"), errors="coerce").fillna(0.0)
            financing = pd.to_numeric(cf.get("Financing Cash Flow"), errors="coerce").fillna(0.0)
            net = pd.to_numeric(cf.get("Net Cash Flow"), errors="coerce").fillna(0.0)
            bridge_delta = (operating + investing + financing - net).abs()
            out["Invariant Cash Flow Bridge Max Delta"] = float(bridge_delta.max()) if not bridge_delta.empty else float("nan")
            out["Invariant Cash Flow Bridge Consistent"] = float(1.0 if (bridge_delta < 1e-3).all() else 0.0)
        else:
            out["Invariant Cash Flow Bridge Max Delta"] = float("nan")
            out["Invariant Cash Flow Bridge Consistent"] = float("nan")

        # 3) Debt opening/closing roll-forward integrity.
        sched = getattr(loan_schedule, "schedule", pd.DataFrame())
        if isinstance(sched, pd.DataFrame) and not sched.empty:
            opening = pd.to_numeric(sched.get("Opening Balance"), errors="coerce").fillna(0.0)
            draw = pd.to_numeric(sched.get("Draw"), errors="coerce").fillna(0.0)
            principal = pd.to_numeric(sched.get("Principal"), errors="coerce").fillna(0.0)
            closing = pd.to_numeric(sched.get("Closing Balance"), errors="coerce").fillna(0.0)
            roll_delta_with_draw = (opening + draw - principal - closing).abs()
            roll_delta_no_draw = (opening - principal - closing).abs()
            with_draw_max = float(roll_delta_with_draw.max()) if not roll_delta_with_draw.empty else float("inf")
            no_draw_max = float(roll_delta_no_draw.max()) if not roll_delta_no_draw.empty else float("inf")
            roll_delta = roll_delta_no_draw if no_draw_max <= with_draw_max else roll_delta_with_draw
            out["Invariant Debt Rollforward Max Delta"] = float(roll_delta.max()) if not roll_delta.empty else float("nan")
            out["Invariant Debt Rollforward Consistent"] = float(1.0 if (roll_delta < 1e-3).all() else 0.0)
        else:
            out["Invariant Debt Rollforward Max Delta"] = float("nan")
            out["Invariant Debt Rollforward Consistent"] = float("nan")

        return out

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

        base_page = copy.deepcopy(self.input_page)
        integrated_mode = self._should_use_integrated_cycle(base_page)
        self._materialize_required_defaults(
            base_page,
            integrated_cycle=integrated_mode,
        )
        farm_capex_sync_audit = (
            self._sync_farm_capex_to_initial_investment(base_page)
            if integrated_mode
            else {"applied": 0, "assets": []}
        )
        page = self._prepare_page_for_scenario(scenario_name, source_page=base_page)
        assumption_audit = self._normalize_global_units(page)
        global_automation_audit = self._auto_balance_global_inputs(page)
        tax_schedule_sync_audit = self._sync_tax_schedule_from_global_rate(page)
        production_annual_sync_audit = (
            {"applied": False, "rows": 0}
            if integrated_mode
            else self._sync_production_annual_from_monthly(page)
        )
        loan_start_sanitize_audit = self._sanitize_loan_schedule_start_month(page)
        debt_envelope_adjustment = self._align_debt_to_capex_envelope(page)
        self._validate_required_inputs(page, integrated_cycle=integrated_mode)
        self._apply_debt_strategy_toggles(page)

        staff_schedule = self._apply_staff_schedule(page)

        projection = page.projection
        depreciation = compute_depreciation_schedule(
            page.initial_investment.model_frame,
            projection.start_year,
            projection.end_year,
        )

        planning_start = projection.planning_start_timestamp

        integrated_cycle_output: IntegratedCycleOutput | None = None
        if integrated_mode:
            integrated_cycle_output = build_integrated_cycle(
                page,
                scenario_name,
                projection.start_year,
                projection.end_year,
                planning_start=planning_start,
            )
            production = ProductionOutput(
                integrated_cycle_output.monthly_physical.copy(),
                integrated_cycle_output.annual_physical.copy(),
            )
            revenue = RevenueOutput(
                integrated_cycle_output.revenue_monthly.copy(),
                integrated_cycle_output.revenue_annual.copy(),
            )

            annual_input = integrated_cycle_output.annual_physical.reset_index()
            annual_input = annual_input.loc[
                annual_input["Year"] >= int(planning_start.year)
            ].copy()
            annual_input["Start Month"] = annual_input["Year"].map(
                lambda year: f"{int(year):04d}-01"
            )
            annual_input = annual_input.rename(
                columns={
                    "Farm Cassava Delivered ton": "Farm Cassava ton",
                    "Purchased Cassava Delivered ton": "Purchased Cassava ton",
                }
            )
            for column in page.production_annual.columns:
                if column not in annual_input.columns:
                    annual_input[column] = 0.0
            page.production_annual.set_data(
                annual_input[page.production_annual.columns],
                mark_user_input=True,
            )
            production_annual_sync_audit = {
                "applied": True,
                "rows": int(len(annual_input)),
                "source": "Integrated annual crop-cycle ledger",
            }

            processed_ton = float(
                integrated_cycle_output.monthly_physical["Cassava Processed ton"].sum()
            )
            feedstock_cost = float(
                integrated_cycle_output.farm_monthly["Total Farm Operating Cost"].sum()
                + integrated_cycle_output.procurement_monthly[
                    "Total Purchased Feedstock Cost"
                ].sum()
            )
            feedstock_sync = {
                "Cassava Feedstock Cost Per Ton (effective)": (
                    feedstock_cost / processed_ton if processed_ton > 0 else 0.0
                )
            }

            direct_inputs = page.direct_costs_monthly.model_frame.copy()
            if not direct_inputs.empty and "Cost Category" in direct_inputs.columns:
                replaced = {"cassava feedstock", "enzymes & chemicals", "energy cost"}
                direct_inputs = direct_inputs.loc[
                    ~direct_inputs["Cost Category"].astype(str).str.strip().str.casefold().isin(replaced)
                ]
            staff_inputs = page.staff_costs_monthly.model_frame.copy()
            if not staff_inputs.empty and "Department" in staff_inputs.columns:
                staff_inputs = staff_inputs.loc[
                    ~staff_inputs["Department"].astype(str).str.contains(
                        "farm", case=False, na=False
                    )
                ]
            other_inputs = page.other_opex_monthly.model_frame.copy()
            if not other_inputs.empty and "Category" in other_inputs.columns:
                replaced_opex = {"energy cost", "sales & marketing"}
                other_inputs = other_inputs.loc[
                    ~other_inputs["Category"].astype(str).str.strip().str.casefold().isin(replaced_opex)
                ]

            cost_outputs = compute_cost_tables(
                direct_inputs,
                staff_inputs,
                other_inputs,
                page.inflation_schedule.model_frame,
                projection.start_year,
                projection.end_year,
                planning_start=planning_start,
            )
            base_direct = cost_outputs["Direct Costs"].monthly.reindex(
                integrated_cycle_output.direct_cost_monthly.index
            ).fillna(0.0)
            integrated_direct = integrated_cycle_output.direct_cost_monthly.copy()
            combined_direct = (
                pd.concat([base_direct, integrated_direct], axis=1)
                .T.groupby(level=0)
                .sum()
                .T
            )
            combined_direct.index.name = "Month"
            combined_direct_annual = combined_direct.groupby(
                combined_direct.index.year
            ).sum()
            combined_direct_annual.index.name = "Year"
            cost_outputs["Direct Costs"] = CostOutput(
                combined_direct,
                combined_direct_annual,
            )
        else:
            production = compute_production_tables(
                page.production_monthly.model_frame,
                projection.start_year,
                projection.end_year,
                planning_start=planning_start,
            )
            feedstock_sync = self._sync_cassava_feedstock_costs(
                page,
                production,
                scenario_name,
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
                planning_start=planning_start,
            )

        loan_schedule = compute_loan_schedule(
            page.loan_schedule.model_frame,
            projection.start_year,
            projection.end_year,
        )

        if integrated_mode and integrated_cycle_output is not None:
            working_capital_monthly = integrated_cycle_output.working_capital_monthly.copy()
            working_capital_annual = working_capital_monthly.groupby(
                working_capital_monthly.index.year
            ).last()
            working_capital_annual.index.name = "Year"
            working_capital = WorkingCapitalOutput(
                working_capital_monthly,
                working_capital_annual,
            )
        else:
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

        risk_commercial = self._apply_risk_and_contract_mechanics(
            page,
            production,
            revenue,
            cost_outputs,
            loan_schedule,
            integrated_cycle=integrated_mode,
        )

        financials = compute_financial_statements(
            revenue,
            depreciation,
            cost_outputs,
            loan_schedule,
            working_capital,
            tax_rate=tax_rate,
            tax_schedule=page.tax_schedule.model_frame,
        )

        debt_covenants = self._apply_dynamic_debt_mechanics(page, loan_schedule, financials)
        if debt_covenants:
            financials = compute_financial_statements(
                revenue,
                depreciation,
                cost_outputs,
                loan_schedule,
                working_capital,
                tax_rate=tax_rate,
                tax_schedule=page.tax_schedule.model_frame,
            )

        expenses: ExpenseSummary = extract_expense_summary(financials, cost_outputs)

        discount_rate = _get_global("Discount rate", 0.0)
        investor_share = _get_global("Investor share capital", 0.0)
        owner_share = _get_global("Owner share capital", float("nan"))
        if investor_share > 1.0 or (np.isfinite(owner_share) and owner_share > 1.0):
            investor_share = investor_share / 100.0 if investor_share > 1.0 else investor_share
            owner_share = owner_share / 100.0 if np.isfinite(owner_share) and owner_share > 1.0 else owner_share
        if not np.isfinite(owner_share):
            owner_share = max(0.0, 1.0 - investor_share)
        share_total = investor_share + owner_share
        if share_total > 0:
            investor_share = investor_share / share_total
            owner_share = owner_share / share_total
        init_df = page.initial_investment.model_frame
        total_investment = (
            float(pd.to_numeric(init_df.get("Cost"), errors="coerce").fillna(0.0).sum())
            if "Cost" in init_df.columns
            else 0.0
        )

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
                "Assumption Quality Checks Passed": float(1.0 if assumption_audit.get("passed", False) else 0.0),
                "Automation Adjustments Applied": float(global_automation_audit.get("applied", 0)),
                "Automation Production Annual Rows": float(production_annual_sync_audit.get("rows", 0)),
                "Automation Tax Schedule Rows Synced": float(tax_schedule_sync_audit.get("rows", 0)),
                **loan_start_sanitize_audit,
                **debt_envelope_adjustment,
                **feedstock_sync,
                **(
                    integrated_cycle_output.metrics
                    if integrated_cycle_output is not None
                    else {"Integrated Cycle Model Enabled": 0.0}
                ),
                **risk_commercial,
                **debt_covenants,
                **self._compute_accounting_invariants(financials, loan_schedule),
            }
        )
        if "Refinancing Costs" not in metrics:
            metrics["Refinancing Costs"] = float(metrics.get("Refinancing Economics Cost", 0.0) or 0.0)
        if not np.isnan(metrics.get("Payback Period (months)", float("nan"))):
            metrics["Payback Period (years)"] = metrics["Payback Period (months)"] / 12.0
        if "Simple Payback (years)" not in metrics:
            metrics["Simple Payback (years)"] = float(metrics.get("Payback Period (years)", float("nan")))

        metrics.update(self._compute_bankability_scorecard(metrics, assumption_audit=assumption_audit))

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
            "integrated_cycle": integrated_cycle_output,
            "staff_schedule": staff_schedule,
            "assumption_quality_audit": assumption_audit,
            "automation_audit": {
                "global_assumptions": global_automation_audit,
                "tax_schedule_sync": tax_schedule_sync_audit,
                "production_annual_sync": production_annual_sync_audit,
                "loan_start_month_sanitize": loan_start_sanitize_audit,
                "farm_capex_sync": farm_capex_sync_audit,
            },
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
