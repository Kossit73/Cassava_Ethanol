from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from .utils import annual_periods, irr, npv, year_month_range


ETHANOL_LITRES_PER_TON = 200.0
ANIMAL_FEED_TON_PER_TON = 0.275


@dataclass
class DepreciationOutput:
    monthly: pd.DataFrame
    annual: pd.DataFrame
    summary: pd.DataFrame
    capex: pd.Series


def compute_depreciation_schedule(initial_investment: pd.DataFrame, start_year: int, end_year: int) -> DepreciationOutput:
    months = year_month_range(start_year, end_year)

    # When the landing-page CAPEX table is still populated with placeholder
    # values ``initial_investment`` can be empty.  The downstream pivot would
    # otherwise raise ``KeyError`` because the required ``Month`` column is
    # missing.  Short-circuit with zero schedules so the rest of the model can
    # continue to build using the user-provided tables only.
    if initial_investment is None or initial_investment.empty:
        empty_monthly = pd.DataFrame(index=months)
        empty_monthly.index.name = "Month"
        empty_monthly["Total Depreciation"] = 0.0
        empty_annual = empty_monthly.resample("Y").sum()
        empty_annual.index = empty_annual.index.year
        empty_summary = pd.DataFrame(columns=[
            "Item",
            "Cost",
            "Life (years)",
            "Depreciation Rate",
            "Annual Depreciation",
            "Monthly Depreciation",
            f"Accumulated Depreciation (Year {end_year})",
            "Net Book Value",
        ])
        capex_series = pd.Series(0.0, index=months, name="Capex")
        return DepreciationOutput(empty_monthly, empty_annual, empty_summary, capex_series)

    records = []
    capex_records = []
    for _, row in initial_investment.iterrows():
        life_years = row.get("Life (years)") or row.get("Life") or 10
        rate = row.get("Depreciation Rate")
        cost = float(row["Cost"])
        if rate in (None, 0, np.nan):
            annual_dep = cost / life_years if life_years else 0
        else:
            annual_dep = cost * float(rate)
        monthly_dep = annual_dep / 12.0
        start_month_value = row.get("Start Month", f"{start_year}-01")
        start_month = pd.Period(start_month_value, freq="M").to_timestamp()
        capex_records.append({"Month": start_month, "Item": row["Item"], "Capex": cost})
        for month in months:
            dep = monthly_dep if month >= start_month else 0.0
            records.append({"Month": month, "Item": row["Item"], "Depreciation": dep})

    monthly_df = (
        pd.DataFrame(records)
        .pivot_table(index="Month", columns="Item", values="Depreciation", aggfunc="sum", fill_value=0.0)
        .sort_index()
    )
    monthly_df["Total Depreciation"] = monthly_df.sum(axis=1)

    capex_df = pd.DataFrame(capex_records)
    if capex_df.empty:
        capex_series = pd.Series(0.0, index=months)
    else:
        capex_series = capex_df.groupby("Month")["Capex"].sum().reindex(months, fill_value=0.0)

    annual_df = monthly_df.resample("Y").sum()
    annual_df.index = annual_df.index.year

    summary = initial_investment.copy()
    summary["Annual Depreciation"] = summary.apply(
        lambda r: (r["Cost"] / r.get("Life (years)") if r.get("Depreciation Rate") in (None, 0, np.nan) else r["Cost"] * r.get("Depreciation Rate")),
        axis=1,
    )
    summary["Monthly Depreciation"] = summary["Annual Depreciation"] / 12.0
    summary["Accumulated Depreciation (Year %d)" % end_year] = summary["Annual Depreciation"] * (end_year - start_year + 1)
    summary["Net Book Value"] = summary["Cost"] - summary["Accumulated Depreciation (Year %d)" % end_year]
    return DepreciationOutput(monthly_df, annual_df, summary, capex_series)


@dataclass
class ProductionOutput:
    monthly: pd.DataFrame
    annual: pd.DataFrame


def compute_production_tables(
    production_monthly: pd.DataFrame,
    start_year: int,
    end_year: int,
    *,
    planning_start: pd.Timestamp | str | pd.Period | None = None,
) -> ProductionOutput:
    """Return compounded cassava volumes and derived ethanol/feed outputs."""

    monthly = production_monthly.copy()
    if monthly.empty:
        empty = pd.DataFrame(columns=["Cassava ton", "Ethanol litres", "Animal Feed ton"])
        empty.index = pd.Index([], name="Month")
        return ProductionOutput(empty, empty)

    month_col = "Month"
    if "Month" not in monthly.columns:
        if "Start Month" in monthly.columns:
            month_col = "Start Month"
        else:
            raise KeyError("Production Monthly table must include a 'Month' or 'Start Month' column")

    month_values = pd.to_datetime(monthly[month_col].astype(str), errors="coerce")
    monthly["Month"] = month_values.dt.to_period("M").dt.to_timestamp()
    monthly = monthly.dropna(subset=["Month"]).sort_values("Month").reset_index(drop=True)
    if month_col != "Month" and month_col in monthly.columns:
        monthly = monthly.drop(columns=[month_col])

    if monthly.empty:
        empty = pd.DataFrame(columns=["Cassava ton", "Ethanol litres", "Animal Feed ton"])
        empty.index = pd.Index([], name="Month")
        return ProductionOutput(empty, empty)

    growth_col = next((c for c in monthly.columns if "growth" in c.lower()), None)
    growth_series = pd.Series(dtype=float)
    if growth_col and growth_col in monthly.columns:
        growth_series = pd.to_numeric(monthly[growth_col], errors="coerce")
        growth_series.index = pd.PeriodIndex(monthly["Month"], freq="M")
        if not growth_series.index.is_unique:
            growth_series = growth_series[~growth_series.index.duplicated(keep="last")]

    monthly = monthly.set_index("Month")
    if growth_col and growth_col in monthly.columns:
        monthly = monthly.drop(columns=[growth_col])

    months = year_month_range(start_year, end_year)
    if months.empty:
        empty_index = pd.DatetimeIndex([], name="Month")
        empty_cols = ["Cassava ton", "Ethanol litres", "Animal Feed ton"]
        empty_monthly = pd.DataFrame(columns=empty_cols, index=empty_index)
        empty_annual = pd.DataFrame(columns=empty_cols)
        return ProductionOutput(empty_monthly, empty_annual)

    if planning_start is not None:
        if isinstance(planning_start, str):
            planning_start_ts = pd.Period(planning_start, freq="M").to_timestamp()
        elif isinstance(planning_start, pd.Period):
            planning_start_ts = planning_start.to_timestamp()
        else:
            planning_start_ts = pd.Timestamp(planning_start)
    else:
        planning_start_ts = None

    cassava_input = pd.to_numeric(monthly.get("Cassava ton", pd.Series(dtype=float)), errors="coerce")
    if isinstance(cassava_input.index, pd.PeriodIndex):
        cassava_input.index = cassava_input.index.to_timestamp()
    cassava_input = cassava_input.dropna()
    if not cassava_input.index.is_unique:
        cassava_input = cassava_input[~cassava_input.index.duplicated(keep="last")]

    growth_lookup: Dict[pd.Timestamp, float] = {}
    if not growth_series.empty:
        growth_lookup = {
            (idx.to_timestamp() if isinstance(idx, pd.Period) else pd.Timestamp(idx)): float(val)
            for idx, val in growth_series.dropna().items()
        }

    base_lookup: Dict[pd.Timestamp, float] = {
        (idx if isinstance(idx, pd.Timestamp) else pd.Timestamp(idx)): float(val)
        for idx, val in cassava_input.items()
    }

    if planning_start_ts is not None:
        base_lookup = {month: val for month, val in base_lookup.items() if month >= planning_start_ts}
        growth_lookup = {month: val for month, val in growth_lookup.items() if month >= planning_start_ts}

    cassava_values = []
    prev_value: float | None = None
    current_growth = 0.0
    for month in months:
        if planning_start_ts is not None and month < planning_start_ts:
            cassava_values.append(0.0)
            prev_value = None
            continue
        if month in growth_lookup:
            new_growth = growth_lookup[month]
            if pd.notna(new_growth):
                current_growth = float(new_growth)
        explicit = base_lookup.get(month)
        if explicit is not None:
            value = explicit
        elif prev_value is not None:
            value = prev_value * (1.0 + current_growth / 12.0)
        else:
            value = 0.0

        cassava_values.append(value)
        if np.isfinite(value):
            prev_value = float(value)

    cassava_series = pd.Series(cassava_values, index=months, name="Cassava ton")
    ethanol_series = cassava_series * ETHANOL_LITRES_PER_TON
    animal_feed_series = cassava_series * ANIMAL_FEED_TON_PER_TON

    compound_monthly = pd.DataFrame(
        {
            "Cassava ton": cassava_series,
            "Ethanol litres": ethanol_series,
            "Animal Feed ton": animal_feed_series,
        }
    )
    compound_monthly.index.name = "Month"

    # Preserve any auxiliary columns (e.g., notes) by forward-filling the
    # user-provided values over the compounded month index.
    auxiliary_cols = [col for col in monthly.columns if col not in compound_monthly.columns]
    for col in auxiliary_cols:
        col_series = monthly[col]
        if isinstance(col_series.index, pd.PeriodIndex):
            col_series.index = col_series.index.to_timestamp()
        compound_monthly[col] = col_series.reindex(months).ffill()

    annual = compound_monthly.resample("Y").sum()
    annual.index = annual.index.year

    start_lookup: Dict[int, str | None] = {}
    for year, group in compound_monthly.groupby(compound_monthly.index.year):
        positive = group[group["Cassava ton"] > 0]
        if not positive.empty:
            start_lookup[year] = positive.index[0].to_period("M").strftime("%Y-%m")
        else:
            start_lookup[year] = None

    annual["Start Month"] = annual.index.map(start_lookup.get)
    ordered_cols = ["Start Month"] + [col for col in annual.columns if col != "Start Month"]
    annual = annual[ordered_cols]
    return ProductionOutput(compound_monthly, annual)


@dataclass
class RevenueOutput:
    monthly: pd.DataFrame
    annual: pd.DataFrame


def compute_revenue_schedule(
    production: ProductionOutput,
    revenue_inputs: pd.DataFrame,
    inflation_schedule: pd.DataFrame,
    *,
    planning_start: pd.Timestamp | str | pd.Period | None = None,
) -> RevenueOutput:
    monthly = production.monthly.copy()
    if not isinstance(monthly.index, pd.DatetimeIndex):
        monthly.index = pd.to_datetime(monthly.index, errors="coerce")
        monthly = monthly[~monthly.index.isna()]
        monthly.index.name = "Month"
    prices = {}
    for _, row in revenue_inputs.iterrows():
        product = row["Product"]
        base_price = row["Base Price"]
        escalation = row.get("Escalation", 0.0)
        prices[product] = (base_price, escalation)

    inflation = inflation_schedule.set_index("Year")["CPI"].to_dict()

    if planning_start is not None:
        if isinstance(planning_start, str):
            planning_start_ts = pd.Period(planning_start, freq="M").to_timestamp()
        elif isinstance(planning_start, pd.Period):
            planning_start_ts = planning_start.to_timestamp()
        else:
            planning_start_ts = pd.Timestamp(planning_start)
    else:
        planning_start_ts = monthly.index[0] if len(monthly.index) else None

    monthly_revenue = pd.DataFrame(index=monthly.index)
    for product, (base_price, escalation) in prices.items():
        product_lower = product.lower()
        if "ethanol" in product_lower:
            volume_col = "Ethanol litres"
        elif any(keyword in product_lower for keyword in ("feed", "anfeed")):
            volume_col = "Animal Feed ton"
        elif "cassava" in product_lower:
            volume_col = "Cassava ton"
        else:
            volume_col = monthly.columns[0]

        if volume_col in monthly.columns:
            volumes = pd.to_numeric(monthly[volume_col], errors="coerce").fillna(0.0)
        else:
            volumes = pd.Series(0.0, index=monthly.index)
        price_series = []
        for ts in monthly.index:
            if planning_start_ts is not None:
                months_since_start = (ts.year - planning_start_ts.year) * 12 + (ts.month - planning_start_ts.month)
                months_since_start = max(0, months_since_start)
            else:
                base_ts = monthly.index[0]
                months_since_start = (ts.year - base_ts.year) * 12 + (ts.month - base_ts.month)
            years_from_start = months_since_start / 12.0
            price = base_price * ((1 + escalation) ** years_from_start)
            cpi = inflation.get(ts.year, 0.0)
            price *= (1 + cpi)
            price_series.append(price)
        monthly_revenue[f"{product} revenue"] = volumes.values * np.array(price_series)
    monthly_revenue["Total Revenue"] = monthly_revenue.sum(axis=1)

    if not isinstance(monthly_revenue.index, pd.DatetimeIndex):
        monthly_revenue.index = pd.to_datetime(monthly_revenue.index, errors="coerce")
        monthly_revenue = monthly_revenue[~monthly_revenue.index.isna()]
        monthly_revenue.index.name = "Month"

    if monthly_revenue.empty:
        annual = pd.DataFrame(columns=monthly_revenue.columns)
    else:
        annual = monthly_revenue.resample("Y").sum()
        annual.index = annual.index.year
    return RevenueOutput(monthly_revenue, annual)


@dataclass
class CostOutput:
    monthly: pd.DataFrame
    annual: pd.DataFrame


@dataclass
class StaffSchedule:
    positions: pd.DataFrame
    department_summary: pd.DataFrame


def compute_staff_schedule(staff_positions: pd.DataFrame) -> StaffSchedule:
    """Return enriched staff position data with monthly/annual cost rollups."""

    columns = ["Position", "Department", "Headcount", "Monthly Salary", "Monthly Cost", "Annual Cost"]
    if staff_positions is None or staff_positions.empty:
        empty_positions = pd.DataFrame(columns=columns)
        empty_summary = pd.DataFrame(
            columns=["Department", "Headcount", "Monthly Cost", "Annual Cost", "Average Monthly Salary"]
        )
        return StaffSchedule(empty_positions, empty_summary)

    df = staff_positions.copy()
    if "Position" not in df.columns:
        df.insert(0, "Position", df.index.astype(str))
    if "Department" not in df.columns:
        df["Department"] = "General"

    df["Headcount"] = pd.to_numeric(df.get("Headcount"), errors="coerce").fillna(0.0)
    df["Monthly Salary"] = pd.to_numeric(df.get("Monthly Salary"), errors="coerce").fillna(0.0)
    df["Monthly Cost"] = df["Headcount"] * df["Monthly Salary"]
    df["Annual Cost"] = df["Monthly Cost"] * 12.0

    summary = (
        df.groupby("Department", dropna=False)[["Headcount", "Monthly Cost", "Annual Cost"]]
        .sum()
        .reset_index()
    )
    summary["Average Monthly Salary"] = summary.apply(
        lambda row: row["Monthly Cost"] / row["Headcount"] if row["Headcount"] else 0.0,
        axis=1,
    )

    ordered_positions = df[columns]
    return StaffSchedule(ordered_positions, summary)


def compute_cost_tables(
    direct_costs: pd.DataFrame,
    staff_costs: pd.DataFrame,
    other_opex: pd.DataFrame,
    inflation_schedule: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> Dict[str, CostOutput]:
    months = year_month_range(start_year, end_year)

    def _prepare(
        df: pd.DataFrame,
        *,
        category_col: str,
        value_column: str,
    ) -> pd.DataFrame:
        """Convert a landing-page table into a monthly cost matrix.

        The landing tables let users rename columns (e.g. "Start Month" versus
        "Month").  The helper therefore locates the first column containing the
        word "month", coerces it to a monthly timestamp, and pivots the chosen
        ``category_col`` against the numeric ``value_column``.  Empty tables
        return an all-zero frame so downstream schedules never raise ``KeyError``
        when the user has not yet supplied inputs.
        """

        if df is None or df.empty:
            return pd.DataFrame(index=months)

        working = df.copy()
        month_col = next((c for c in working.columns if "month" in c.lower()), None)
        if not month_col:
            return pd.DataFrame(index=months)

        month_values = pd.to_datetime(working[month_col].astype(str), errors="coerce")
        working = working.loc[month_values.notna()].copy()
        if working.empty:
            return pd.DataFrame(index=months)

        working["Month"] = month_values.loc[working.index].dt.to_period("M").dt.to_timestamp()
        working[value_column] = pd.to_numeric(working[value_column], errors="coerce").fillna(0.0)
        working[category_col] = working[category_col].astype(str).fillna("Category")

        pivot = (
            working.pivot_table(
                index="Month",
                columns=category_col,
                values=value_column,
                aggfunc="sum",
            )
            .sort_index()
            .reindex(months)
        )

        # Forward-fill within the projection horizon so dated overrides continue
        # to apply after their effective month.
        return pivot.ffill().fillna(0.0)

    direct = _prepare(direct_costs, category_col="Cost Category", value_column="Amount")
    staff = _prepare(staff_costs, category_col="Department", value_column="Cost")
    other = _prepare(other_opex, category_col="Category", value_column="Amount")

    outputs = {}
    for name, table in {
        "Direct Costs": direct,
        "Staff Costs": staff,
        "Other Opex": other,
    }.items():
        if table.empty:
            annual = pd.DataFrame(columns=[])
        else:
            annual = table.resample("YE").sum()
            annual.index = annual.index.year
        outputs[name] = CostOutput(table, annual)
    return outputs


@dataclass
class LoanScheduleOutput:
    schedule: pd.DataFrame
    summary: pd.DataFrame
    annual: pd.DataFrame


def compute_loan_schedule(
    loan_inputs: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> LoanScheduleOutput:
    months = year_month_range(start_year, end_year)
    schedule_rows = []

    if loan_inputs is None or loan_inputs.empty:
        empty_schedule = pd.DataFrame(
            columns=[
                "Loan",
                "Month",
                "Draw",
                "Opening Balance",
                "Interest",
                "Principal",
                "Closing Balance",
                "Payment",
            ]
        )
        return LoanScheduleOutput(
            empty_schedule,
            pd.DataFrame(columns=["Draw", "Interest", "Principal", "Payment"]),
            pd.DataFrame(
                columns=[
                    "Loan",
                    "Year",
                    "Interest Rate",
                    "Yearly Remaining Balance",
                    "Monthly Interest (Balance × Rate / 12)",
                    "Interest Paid",
                    "Principal Paid",
                    "Total Payment",
                    "Year-End Balance",
                ]
            ),
        )

    amount_columns = ["Loan Amount", "Amount", "Drawdown"]

    def _normalise_interest_rate(value: float | str | None) -> float:
        """Return a decimal interest rate from user input.

        The landing page captures rates as percentages (``7.5`` or ``"7.5%"``)
        while tests feed decimals (``0.075``).  Streamline both by treating
        values greater than ``1`` as percentages and gracefully handling
        strings with a trailing percent sign.
        """

        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return 0.0

        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned.endswith("%"):
                cleaned = cleaned[:-1]
            try:
                numeric = float(cleaned)
            except ValueError:
                return 0.0
        else:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return 0.0

        if not np.isfinite(numeric):
            return 0.0

        return numeric / 100.0 if abs(numeric) > 1 else numeric

    for _, loan in loan_inputs.iterrows():
        amount = None
        for column in amount_columns:
            if column in loan.index:
                try:
                    value = float(loan.get(column))
                except (TypeError, ValueError):
                    value = float("nan")
                if np.isfinite(value) and value > 0:
                    amount = value
                    break
        if amount is None:
            continue

        tenor_years = int(loan.get("Tenor Years", 8))
        grace_years = int(loan.get("Grace Years", 1))
        rate = _normalise_interest_rate(loan.get("Interest Rate", 0.08))
        amortization = str(loan.get("Amortization", "Annuity") or "Annuity")
        monthly_rate = rate / 12.0
        tenor_months = max(tenor_years, 0) * 12
        grace_months = max(grace_years, 0) * 12
        repay_months = max(tenor_months - grace_months, 0)

        draw_month_value = loan.get("Start Month") or months[0]
        draw_month = pd.Period(draw_month_value, freq="M").to_timestamp()
        drawn = False
        payments_made = 0
        balance = 0.0
        annuity_payment = 0.0
        if amortization.lower().startswith("ann") and repay_months > 0:
            if monthly_rate == 0:
                annuity_payment = amount / repay_months
            else:
                factor = (monthly_rate * (1 + monthly_rate) ** repay_months) / (
                    (1 + monthly_rate) ** repay_months - 1
                )
                annuity_payment = amount * factor
        elif repay_months > 0:
            annuity_payment = amount / repay_months

        for month in months:
            draw = 0.0
            if not drawn and month >= draw_month:
                draw = amount
                balance += draw
                drawn = True

            opening_balance = balance
            interest = 0.0
            principal = 0.0
            payment = 0.0

            if drawn:
                months_since_draw = (month.year - draw_month.year) * 12 + (month.month - draw_month.month)
                if months_since_draw < grace_months:
                    interest = opening_balance * monthly_rate
                elif payments_made < repay_months and opening_balance > 0:
                    interest = opening_balance * monthly_rate
                    payment = annuity_payment
                    if amortization.lower().startswith("ann"):
                        principal = max(0.0, payment - interest)
                    else:
                        principal = annuity_payment
                        payment = interest + principal
                    principal = min(principal, opening_balance)
                    balance = max(0.0, opening_balance - principal)
                    payments_made += 1
                else:
                    balance = opening_balance

            schedule_rows.append(
                {
                    "Loan": loan.get("Loan") or "Loan",
                    "Month": month,
                    "Draw": draw,
                    "Opening Balance": opening_balance,
                    "Interest": interest,
                    "Principal": principal,
                    "Closing Balance": balance,
                    "Payment": interest + principal,
                    "Interest Rate": rate,
                }
            )

    schedule = pd.DataFrame(schedule_rows)
    if schedule.empty:
        summary = pd.DataFrame(columns=["Draw", "Interest", "Principal", "Payment"])
        annual = pd.DataFrame(
            columns=[
                "Loan",
                "Year",
                "Interest Rate",
                "Yearly Remaining Balance",
                "Monthly Interest (Balance × Rate / 12)",
                "Interest Paid",
                "Principal Paid",
                "Total Payment",
                "Year-End Balance",
            ]
        )
    else:
        schedule = schedule.sort_values(["Loan", "Month"]).reset_index(drop=True)
        schedule["Month"] = pd.to_datetime(schedule["Month"], errors="coerce")
        summary = schedule.groupby("Loan").agg({"Draw": "sum", "Interest": "sum", "Principal": "sum", "Payment": "sum"})
        schedule["Year"] = schedule["Month"].dt.year
        annual = (
            schedule.groupby(["Loan", "Year"]).agg(
                Interest_Rate=("Interest Rate", "first"),
                Yearly_Remaining_Balance=("Opening Balance", "first"),
                Interest_Paid=("Interest", "sum"),
                Principal_Paid=("Principal", "sum"),
                Total_Payment=("Payment", "sum"),
                Year_End_Balance=("Closing Balance", "last"),
            )
            .reset_index()
            .sort_values(["Loan", "Year"])
        )
        annual["Monthly Interest (Balance × Rate / 12)"] = (
            annual["Yearly_Remaining_Balance"] * annual["Interest_Rate"] / 12.0
        )
        annual = annual.rename(
            columns={
                "Interest_Rate": "Interest Rate",
                "Yearly_Remaining_Balance": "Yearly Remaining Balance",
                "Interest_Paid": "Interest Paid",
                "Principal_Paid": "Principal Paid",
                "Total_Payment": "Total Payment",
                "Year_End_Balance": "Year-End Balance",
            }
        )
        annual = annual[
            [
                "Loan",
                "Year",
                "Interest Rate",
                "Yearly Remaining Balance",
                "Monthly Interest (Balance × Rate / 12)",
                "Interest Paid",
                "Principal Paid",
                "Total Payment",
                "Year-End Balance",
            ]
        ]
    return LoanScheduleOutput(schedule, summary, annual)


@dataclass
class WorkingCapitalOutput:
    monthly: pd.DataFrame
    annual: pd.DataFrame


def compute_working_capital(
    revenue: RevenueOutput,
    cost_outputs: Dict[str, CostOutput],
    accounts_receivable_inputs: pd.DataFrame,
    inventory_inputs: pd.DataFrame,
) -> WorkingCapitalOutput:
    if "Total Revenue" in revenue.monthly.columns:
        monthly_revenue = revenue.monthly["Total Revenue"]
    else:
        months_index = revenue.monthly.index
        if months_index.empty:
            empty = pd.DataFrame(
                columns=
                [
                    "Receivables",
                    "Inventory",
                    "Prepaid Expenses",
                    "Other Assets",
                    "Payables",
                    "Other Payables",
                    "Net Working Capital",
                ]
            )
            return WorkingCapitalOutput(empty, pd.DataFrame())
        monthly_revenue = pd.Series(0.0, index=months_index)

    months = monthly_revenue.index

    def _sum_cost(name: str) -> pd.Series:
        output = cost_outputs.get(name)
        if output is None or output.monthly.empty:
            return pd.Series(0.0, index=months)
        reindexed = output.monthly.reindex(months).fillna(0.0)
        return reindexed.sum(axis=1)

    direct_costs = _sum_cost("Direct Costs")
    staff_costs = _sum_cost("Staff Costs")
    other_costs = _sum_cost("Other Opex")
    monthly_cogs = direct_costs + staff_costs + other_costs
    operating_costs = staff_costs + other_costs

    def _metric_series(
        df: pd.DataFrame, metric_name: str, default: float = 0.0
    ) -> Tuple[pd.Series, bool]:
        series = pd.Series(default, index=months, dtype=float)
        if df is None or df.empty or "Metric" not in df.columns:
            return series, False

        working = df.copy()
        effective_col = next((c for c in ("Effective Month", "Start Month", "Month") if c in working.columns), None)
        if effective_col is None:
            working["Effective Month"] = months[0]
            effective_col = "Effective Month"

        effective_dates = pd.to_datetime(working[effective_col].astype(str), errors="coerce")
        working["_effective"] = effective_dates.dt.to_period("M").dt.to_timestamp()
        working["_metric"] = working["Metric"].astype(str).str.lower()
        working["_value"] = pd.to_numeric(working.get("Value"), errors="coerce")
        mask = (
            working["_metric"] == metric_name.lower()
        ) & working["_effective"].notna() & working["_value"].notna()
        filtered = working.loc[mask, ["_effective", "_value"]].sort_values("_effective")
        if filtered.empty:
            return series, False

        current_value = default
        idx = 0
        effective_values = filtered.values.tolist()
        for month in months:
            while idx < len(effective_values) and effective_values[idx][0] <= month:
                current_value = float(effective_values[idx][1])
                idx += 1
            series.loc[month] = current_value
        return series, True

    ar_days_series, _ = _metric_series(accounts_receivable_inputs, "Receivables days", 0.0)
    inventory_days_series, has_inventory = _metric_series(accounts_receivable_inputs, "Inventory days", 0.0)
    if not has_inventory:
        inventory_days_series, _ = _metric_series(inventory_inputs, "Inventory days", 0.0)

    prepaid_days_series, has_prepaid = _metric_series(accounts_receivable_inputs, "Prepaid expense days", 0.0)
    other_asset_pct_series, has_other_assets = _metric_series(
        accounts_receivable_inputs, "Other assets percent of revenue", 0.0
    )
    payables_days_series, has_payables = _metric_series(accounts_receivable_inputs, "Payables days", 0.0)
    if not has_payables:
        payables_days_series, _ = _metric_series(inventory_inputs, "Payables days", 0.0)
    other_payable_days_series, has_other_payables = _metric_series(
        accounts_receivable_inputs, "Other payable days", 0.0
    )
    if not has_other_payables:
        other_payable_days_series, _ = _metric_series(inventory_inputs, "Other payable days", 0.0)

    receivables = monthly_revenue * (ar_days_series / 30.0)
    inventory = monthly_cogs * (inventory_days_series / 30.0)
    prepaid = operating_costs * (prepaid_days_series / 30.0)
    other_assets = monthly_revenue * other_asset_pct_series
    payables = monthly_cogs * (payables_days_series / 30.0)
    other_payables = operating_costs * (other_payable_days_series / 30.0)

    wc = pd.DataFrame(
        {
            "Receivables": receivables,
            "Inventory": inventory,
            "Prepaid Expenses": prepaid,
            "Other Assets": other_assets,
            "Payables": payables,
            "Other Payables": other_payables,
        }
    )
    wc["Net Working Capital"] = (
        receivables
        + inventory
        + prepaid
        + other_assets
        - payables
        - other_payables
    )

    if wc.empty:
        annual = pd.DataFrame(columns=wc.columns)
    else:
        annual = wc.resample("Y").last()
        annual.index = annual.index.year
        start_year = int(months.min().year)
        end_year = int(months.max().year)
        year_index = pd.Index(range(start_year, end_year + 1), name="Year")
        annual = annual.reindex(year_index, fill_value=0.0)
    return WorkingCapitalOutput(wc, annual)


@dataclass
class FinancialStatements:
    income_monthly: pd.DataFrame
    income_annual: pd.DataFrame
    balance_monthly: pd.DataFrame
    balance_annual: pd.DataFrame
    cashflow_monthly: pd.DataFrame
    cashflow_annual: pd.DataFrame
    balance_ratios_monthly: pd.DataFrame = field(default_factory=pd.DataFrame)
    balance_ratios_annual: pd.DataFrame = field(default_factory=pd.DataFrame)
    income_ratios_monthly: pd.DataFrame = field(default_factory=pd.DataFrame)
    income_ratios_annual: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class ExpenseSummary:
    """Container for the key expense lines used across dashboards."""

    monthly: pd.DataFrame
    annual: pd.DataFrame


def compute_financial_statements(
    revenue: RevenueOutput,
    depreciation: DepreciationOutput,
    cost_outputs: Dict[str, CostOutput],
    loan_schedule: LoanScheduleOutput,
    working_capital: WorkingCapitalOutput,
    tax_rate: float,
) -> FinancialStatements:
    revenue_monthly = revenue.monthly.copy()
    if not isinstance(revenue_monthly.index, pd.DatetimeIndex):
        revenue_monthly.index = pd.to_datetime(revenue_monthly.index, errors="coerce")
        revenue_monthly = revenue_monthly[~revenue_monthly.index.isna()]
    candidate_indexes = []
    if not revenue_monthly.empty:
        candidate_indexes.append(revenue_monthly.index)
    for output in cost_outputs.values():
        idx = getattr(output.monthly, "index", None)
        if isinstance(idx, pd.DatetimeIndex) and len(idx):
            candidate_indexes.append(idx)
    dep_index = getattr(depreciation.monthly, "index", None)
    if isinstance(dep_index, pd.DatetimeIndex) and len(dep_index):
        candidate_indexes.append(dep_index)
    wc_index = getattr(getattr(working_capital, "monthly", None), "index", None)
    if isinstance(wc_index, pd.DatetimeIndex) and len(wc_index):
        candidate_indexes.append(wc_index)

    if candidate_indexes:
        monthly_index = candidate_indexes[0]
        for idx in candidate_indexes[1:]:
            monthly_index = monthly_index.union(idx)
        monthly_index = monthly_index.sort_values()
    else:
        monthly_index = pd.DatetimeIndex([], name="Month")

    monthly = revenue_monthly.reindex(monthly_index, fill_value=0.0)
    monthly.index.name = "Month"

    dep_source = depreciation.monthly.get("Total Depreciation") if hasattr(depreciation.monthly, "get") else None
    if dep_source is None:
        dep = pd.Series(0.0, index=monthly_index)
    else:
        dep = pd.to_numeric(dep_source, errors="coerce").reindex(monthly_index, fill_value=0.0)

    def _cost_series(name: str) -> pd.Series:
        output = cost_outputs.get(name)
        if output is None or output.monthly.empty:
            return pd.Series(0.0, index=monthly_index)
        table = output.monthly
        if not isinstance(table.index, pd.DatetimeIndex):
            table = table.copy()
            table.index = pd.to_datetime(table.index, errors="coerce")
            table = table[~table.index.isna()]
        return table.reindex(monthly_index, fill_value=0.0).sum(axis=1)

    direct = _cost_series("Direct Costs")
    staff = _cost_series("Staff Costs")
    other = _cost_series("Other Opex")
    schedule_df = loan_schedule.schedule
    if schedule_df.empty:
        interest = pd.Series(0.0, index=monthly_index)
        debt_draws = pd.Series(0.0, index=monthly_index)
        principal = pd.Series(0.0, index=monthly_index)
        closing_balance = pd.Series(0.0, index=monthly_index)
    else:
        schedule_df = schedule_df.copy()
        if not isinstance(schedule_df.index, pd.DatetimeIndex):
            schedule_df = schedule_df.set_index("Month")
        interest = (
            schedule_df.groupby(level=0)["Interest"].sum()
            .reindex(monthly_index, fill_value=0.0)
        )
        if "Draw" in schedule_df.columns:
            debt_draws = (
                schedule_df.groupby(level=0)["Draw"].sum()
                .reindex(monthly_index, fill_value=0.0)
            )
        else:
            debt_draws = pd.Series(0.0, index=monthly_index)
        principal = (
            schedule_df.groupby(level=0)["Principal"].sum()
            .reindex(monthly_index, fill_value=0.0)
        )
        closing_balance = (
            schedule_df.groupby(level=0)["Closing Balance"].sum()
            .reindex(monthly_index, fill_value=0.0)
        )
    debt_service = principal + interest
    capex = depreciation.capex.reindex(monthly.index, fill_value=0.0)

    income_monthly = pd.DataFrame(index=monthly.index)
    total_revenue = monthly.get("Total Revenue")
    if total_revenue is None:
        total_revenue = pd.Series(0.0, index=monthly.index)
    else:
        total_revenue = total_revenue.reindex(monthly.index, fill_value=0.0)
    income_monthly["Revenue"] = total_revenue
    income_monthly["COGS"] = direct
    income_monthly["Staff Costs"] = staff
    income_monthly["Other Opex"] = other
    income_monthly["EBITDA"] = income_monthly["Revenue"] - income_monthly[["COGS", "Staff Costs", "Other Opex"]].sum(axis=1)
    income_monthly["Depreciation"] = dep
    income_monthly["EBIT"] = income_monthly["EBITDA"] - dep
    income_monthly["Interest"] = interest
    income_monthly["EBT"] = income_monthly["EBIT"] - interest
    income_monthly["Tax"] = income_monthly["EBT"].clip(lower=0) * tax_rate
    income_monthly["Net Income"] = income_monthly["EBT"] - income_monthly["Tax"]

    income_annual = income_monthly.resample("Y").sum()
    income_annual.index = income_annual.index.year

    wc_monthly = working_capital.monthly
    if wc_monthly is None or wc_monthly.empty:
        wc_monthly = pd.DataFrame(index=monthly.index)
    wc_monthly = wc_monthly.reindex(monthly.index).fillna(0.0)

    def _wc_series(column: str, default: float = 0.0) -> pd.Series:
        series = wc_monthly.get(column)
        if series is None:
            return pd.Series(default, index=monthly.index, dtype=float)
        return pd.to_numeric(series, errors="coerce").reindex(monthly.index, fill_value=default)

    receivables = _wc_series("Receivables")
    inventory = _wc_series("Inventory")
    prepaid = _wc_series("Prepaid Expenses")
    other_assets = _wc_series("Other Assets")
    payables = _wc_series("Payables")
    other_payables = _wc_series("Other Payables")

    accounts_receivable_other = (
        receivables.add(prepaid, fill_value=0.0).add(other_assets, fill_value=0.0)
    )
    net_working_capital = wc_monthly.get("Net Working Capital")
    if net_working_capital is None:
        net_working_capital = (
            accounts_receivable_other + inventory - payables - other_payables
        )
    else:
        net_working_capital = pd.to_numeric(
            net_working_capital, errors="coerce"
        ).reindex(monthly.index, fill_value=0.0)

    delta_wc = net_working_capital.diff().fillna(net_working_capital)

    cashflow_monthly = pd.DataFrame(index=monthly.index)
    cashflow_monthly["Net Income"] = income_monthly["Net Income"]
    cashflow_monthly["Depreciation"] = dep
    cashflow_monthly["Operating Cash Flow"] = income_monthly["Net Income"] + dep - delta_wc
    cashflow_monthly["Capex"] = capex
    cashflow_monthly["Investing Cash Flow"] = -capex
    cashflow_monthly["Free Cash Flow"] = cashflow_monthly["Operating Cash Flow"] - capex
    cashflow_monthly["Debt Draws"] = debt_draws
    cashflow_monthly["Debt Service"] = debt_service
    cashflow_monthly["Financing Cash Flow"] = debt_draws - debt_service
    cashflow_monthly["Equity Cash Flow"] = cashflow_monthly["Free Cash Flow"] + cashflow_monthly["Financing Cash Flow"]
    cashflow_monthly["Net Cash Flow"] = (
        cashflow_monthly["Operating Cash Flow"]
        + cashflow_monthly["Investing Cash Flow"]
        + cashflow_monthly["Financing Cash Flow"]
    )

    cashflow_annual = cashflow_monthly.resample("Y").sum()
    cashflow_annual.index = cashflow_annual.index.year

    balance_monthly = pd.DataFrame(index=monthly.index)
    balance_monthly["Cash"] = cashflow_monthly["Net Cash Flow"].cumsum()
    gross_ppe = capex.cumsum()
    accumulated_dep = dep.cumsum()
    balance_monthly["Net PP&E"] = gross_ppe - accumulated_dep
    balance_monthly["Accounts Receivable & Other Assets"] = accounts_receivable_other
    balance_monthly["Inventory"] = inventory
    balance_monthly["Accounts Payable"] = payables
    balance_monthly["Other Payables"] = other_payables
    balance_monthly["Debt"] = closing_balance

    asset_columns = [
        "Cash",
        "Accounts Receivable & Other Assets",
        "Inventory",
        "Net PP&E",
    ]
    liability_columns = ["Accounts Payable", "Other Payables", "Debt"]
    total_assets = balance_monthly[asset_columns].sum(axis=1)
    total_liabilities = balance_monthly[liability_columns].sum(axis=1)
    balance_monthly["Equity"] = total_assets - total_liabilities
    balance_monthly["Total Assets"] = total_assets
    balance_monthly["Total Liabilities"] = total_liabilities
    balance_monthly["Total Liabilities & Equity"] = total_liabilities + balance_monthly["Equity"]

    balance_annual = balance_monthly.resample("Y").last()
    balance_annual.index = balance_annual.index.year

    def _balance_series(column: str) -> pd.Series:
        if column in balance_monthly.columns:
            return pd.to_numeric(balance_monthly[column], errors="coerce").reindex(balance_monthly.index, fill_value=0.0)
        return pd.Series(0.0, index=balance_monthly.index)

    def _income_series(column: str) -> pd.Series:
        if column in income_monthly.columns:
            return pd.to_numeric(income_monthly[column], errors="coerce").reindex(income_monthly.index, fill_value=0.0)
        return pd.Series(0.0, index=income_monthly.index)

    def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        denominator = denominator.where(denominator != 0, np.nan)
        result = numerator.divide(denominator)
        return result.replace([np.inf, -np.inf], np.nan)

    cash_series = _balance_series("Cash")
    receivables_series = _balance_series("Accounts Receivable & Other Assets")
    inventory_series = _balance_series("Inventory")
    current_assets = cash_series + receivables_series + inventory_series
    current_liabilities = (
        _balance_series("Accounts Payable")
        + _balance_series("Other Payables")
        + _balance_series("Debt")
    )
    total_liabilities = _balance_series("Total Liabilities")
    total_assets = _balance_series("Total Assets")
    equity_series = _balance_series("Equity")

    balance_ratios_monthly = pd.DataFrame(index=balance_monthly.index)
    balance_ratios_monthly["Current Ratio"] = _safe_ratio(current_assets, current_liabilities)
    balance_ratios_monthly["Debt-to-Equity Ratio"] = _safe_ratio(total_liabilities, equity_series)
    balance_ratios_monthly["Debt Ratio"] = _safe_ratio(total_liabilities, total_assets)

    balance_ratios_annual = balance_ratios_monthly.resample("Y").last()
    balance_ratios_annual.index = balance_ratios_annual.index.year

    revenue_series = _income_series("Revenue")
    cogs_series = _income_series("COGS")
    net_income_series = _income_series("Net Income")
    gross_profit_series = revenue_series - cogs_series
    average_equity = (equity_series + equity_series.shift(1)).div(2).fillna(equity_series)

    income_ratios_monthly = pd.DataFrame(index=income_monthly.index)
    income_ratios_monthly["Gross Margin"] = _safe_ratio(gross_profit_series, revenue_series)
    income_ratios_monthly["Return on Assets (ROA)"] = _safe_ratio(net_income_series, total_assets)
    income_ratios_monthly["Return on Equity (ROE)"] = _safe_ratio(net_income_series, average_equity)

    income_ratios_annual = pd.DataFrame(index=income_annual.index)
    if not income_annual.empty:
        annual_revenue = pd.to_numeric(income_annual.get("Revenue", pd.Series(0.0, index=income_annual.index)), errors="coerce").fillna(0.0)
        annual_cogs = pd.to_numeric(income_annual.get("COGS", pd.Series(0.0, index=income_annual.index)), errors="coerce").fillna(0.0)
        annual_net_income = pd.to_numeric(income_annual.get("Net Income", pd.Series(0.0, index=income_annual.index)), errors="coerce").fillna(0.0)
        annual_gross_profit = annual_revenue - annual_cogs
        annual_assets = pd.to_numeric(balance_annual.get("Total Assets", pd.Series(0.0, index=balance_annual.index)), errors="coerce").fillna(0.0)
        annual_equity = pd.to_numeric(balance_annual.get("Equity", pd.Series(0.0, index=balance_annual.index)), errors="coerce").fillna(0.0)
        annual_avg_equity = (annual_equity + annual_equity.shift(1)).div(2).fillna(annual_equity)

        income_ratios_annual["Gross Margin"] = _safe_ratio(annual_gross_profit, annual_revenue)
        income_ratios_annual["Return on Assets (ROA)"] = _safe_ratio(annual_net_income, annual_assets)
        income_ratios_annual["Return on Equity (ROE)"] = _safe_ratio(annual_net_income, annual_avg_equity)

    return FinancialStatements(
        income_monthly=income_monthly,
        income_annual=income_annual,
        balance_monthly=balance_monthly,
        balance_annual=balance_annual,
        cashflow_monthly=cashflow_monthly,
        cashflow_annual=cashflow_annual,
        balance_ratios_monthly=balance_ratios_monthly,
        balance_ratios_annual=balance_ratios_annual,
        income_ratios_monthly=income_ratios_monthly,
        income_ratios_annual=income_ratios_annual,
    )


def extract_expense_summary(
    financials: FinancialStatements,
    cost_outputs: Dict[str, CostOutput] | None = None,
    expense_columns: Iterable[str] = ("COGS", "Staff Costs", "Other Opex", "Tax"),
) -> ExpenseSummary:
    """Return the headline expense lines with fallbacks to cost schedules.

    When the income statement already contains the requested columns they take
    precedence.  Otherwise the helper derives the series from the underlying
    cost tables so that edited landing-page inputs continue to flow through the
    downstream dashboards even if the financial statement has not yet been
    populated (e.g. when revenue is still zero).  The returned dataframes always
    contain the requested ``expense_columns`` in a consistent order.
    """

    expense_columns = tuple(expense_columns)
    cost_lookup = {
        "COGS": "Direct Costs",
        "Staff Costs": "Staff Costs",
        "Other Opex": "Other Opex",
    }

    def _monthly_index() -> pd.DatetimeIndex:
        idx = getattr(financials.income_monthly, "index", None)
        if isinstance(idx, pd.DatetimeIndex) and len(idx):
            return idx
        candidates: list[pd.DatetimeIndex] = []
        if cost_outputs:
            for output in cost_outputs.values():
                output_idx = getattr(output.monthly, "index", None)
                if isinstance(output_idx, pd.DatetimeIndex) and len(output_idx):
                    candidates.append(output_idx)
        if candidates:
            combined = candidates[0]
            for other in candidates[1:]:
                combined = combined.union(other)
            return combined.sort_values()
        return pd.DatetimeIndex([], name="Month")

    monthly_index = _monthly_index()
    monthly_df = pd.DataFrame(index=monthly_index, columns=expense_columns, dtype=float)
    for column in expense_columns:
        series = None
        if column in financials.income_monthly.columns:
            series = pd.to_numeric(financials.income_monthly[column], errors="coerce")
        elif cost_outputs and column in cost_lookup:
            output = cost_outputs.get(cost_lookup[column])
            if output is not None and not output.monthly.empty:
                table = output.monthly
                if not isinstance(table.index, pd.DatetimeIndex):
                    table = table.copy()
                    table.index = pd.to_datetime(table.index, errors="coerce")
                    table = table[~table.index.isna()]
                series = table.sum(axis=1)
        if series is None:
            series = pd.Series(0.0, index=monthly_index)
        monthly_df[column] = series.reindex(monthly_index, fill_value=0.0)

    monthly_df = monthly_df.fillna(0.0)
    if monthly_df.index.name is None:
        monthly_df.index.name = "Month"

    if not monthly_df.empty:
        annual_df = monthly_df.resample("Y").sum()
        annual_df.index = annual_df.index.year
    else:
        annual_df = pd.DataFrame(columns=expense_columns)

    if "Tax" in financials.income_annual.columns and not financials.income_annual.empty:
        tax_series = pd.to_numeric(financials.income_annual["Tax"], errors="coerce")
        if annual_df.empty:
            annual_df = pd.DataFrame(index=tax_series.index, columns=expense_columns, dtype=float)
        annual_df["Tax"] = tax_series.reindex(annual_df.index, fill_value=0.0)

    if not annual_df.empty:
        annual_df = annual_df.fillna(0.0)
        annual_df.index.name = "Year"

    return ExpenseSummary(monthly=monthly_df, annual=annual_df)


def _derive_initial_investment_components(
    financials: FinancialStatements,
    revenue: RevenueOutput | None = None,
) -> Tuple[float, float, pd.Series, pd.Series]:
    """Return the pre-operation investment split and adjusted cash flows."""

    cashflow_monthly = financials.cashflow_monthly
    free_cash_flow = pd.to_numeric(
        cashflow_monthly.get("Free Cash Flow"), errors="coerce"
    ).fillna(0.0)
    financing_cash_flow = pd.to_numeric(
        cashflow_monthly.get("Financing Cash Flow"), errors="coerce"
    ).fillna(0.0)
    capex_series = pd.to_numeric(cashflow_monthly.get("Capex"), errors="coerce").fillna(0.0)
    loan_draws = pd.to_numeric(cashflow_monthly.get("Debt Draws"), errors="coerce").fillna(0.0)

    index = free_cash_flow.index
    operations_start: pd.Timestamp | None = None

    if revenue is not None and hasattr(revenue, "monthly"):
        revenue_monthly = revenue.monthly
        if isinstance(revenue_monthly, pd.DataFrame) and not revenue_monthly.empty:
            monthly = revenue_monthly.copy()
            if not isinstance(monthly.index, pd.DatetimeIndex):
                monthly.index = pd.to_datetime(monthly.index, errors="coerce")
                monthly = monthly[~monthly.index.isna()]
            total_revenue = pd.to_numeric(monthly.get("Total Revenue"), errors="coerce").fillna(0.0)
            positive_revenue = total_revenue[total_revenue > 1e-9]
            if not positive_revenue.empty:
                operations_start = positive_revenue.index.min()

    if operations_start is None:
        operating_cf = pd.to_numeric(
            cashflow_monthly.get("Operating Cash Flow"), errors="coerce"
        ).fillna(0.0)
        positive_operating = operating_cf[operating_cf > 1e-9]
        if not positive_operating.empty:
            operations_start = positive_operating.index.min()

    initial_mask = pd.Series(False, index=index)
    if operations_start is not None:
        initial_mask.loc[index < operations_start] = True
    if not initial_mask.any() and len(index) > 0:
        initial_mask.iloc[0] = True

    initial_capex = capex_series.where(initial_mask, 0.0)
    initial_loan = loan_draws.where(initial_mask, 0.0)

    adjusted_free_cash_flow = free_cash_flow + initial_capex
    adjusted_financing_cash_flow = financing_cash_flow - initial_loan

    total_initial_investment = float(initial_capex.sum())
    total_initial_loan = float(initial_loan.sum())
    return (
        total_initial_investment,
        total_initial_loan,
        adjusted_free_cash_flow,
        adjusted_financing_cash_flow,
    )


def compute_key_metrics(
    financials: FinancialStatements,
    discount_rate: float,
    investor_share: float,
    owner_share: float,
    *,
    revenue: RevenueOutput | None = None,
) -> Dict[str, float]:
    raw_free_cash_flow = financials.cashflow_monthly["Free Cash Flow"].astype(float)
    (
        initial_project_outlay,
        initial_loan_draw,
        adjusted_free_cash_flow,
        adjusted_financing_cash_flow,
    ) = _derive_initial_investment_components(financials, revenue)
    equity_cash_flow = adjusted_free_cash_flow + adjusted_financing_cash_flow

    def _cumulative_total(series: pd.Series) -> float:
        if series.empty:
            return 0.0
        return float(series.cumsum().iloc[-1])

    def _final_value(series: pd.Series) -> float:
        if series.empty:
            return 0.0
        return float(series.iloc[-1])

    initial_equity = initial_project_outlay - initial_loan_draw

    project_cashflows = [-initial_project_outlay] + adjusted_free_cash_flow.tolist()
    equity_cashflows = [-initial_equity] + equity_cash_flow.tolist()
    investor_cashflows = [-initial_equity * investor_share] + (
        equity_cash_flow * investor_share
    ).tolist()
    owner_cashflows = [-initial_equity * owner_share] + (
        equity_cash_flow * owner_share
    ).tolist()

    project_npv = npv(discount_rate / 12, project_cashflows)
    project_irr = irr(project_cashflows)
    equity_irr = irr(equity_cashflows)
    investor_irr = irr(investor_cashflows)
    owner_irr = irr(owner_cashflows)

    cumulative_project = np.cumsum(project_cashflows)
    payback_months = float("nan")
    payback_label = None
    if np.any(cumulative_project[1:] >= 0):
        crossing = int(np.argmax(cumulative_project[1:] >= 0)) + 1
        prev_cum = cumulative_project[crossing - 1]
        period_cf = project_cashflows[crossing]
        fraction = (-prev_cum / period_cf) if period_cf != 0 else 0.0
        payback_months = crossing - 1 + fraction
        if crossing - 1 < len(financials.cashflow_monthly.index):
            payback_date = financials.cashflow_monthly.index[crossing - 1]
            payback_label = payback_date.strftime("%Y-%m")

    metrics = {
        "Project NPV": project_npv,
        "Project IRR": project_irr,
        "Equity IRR": equity_irr,
        "Investor IRR": investor_irr,
        "Owner IRR": owner_irr,
        "Initial Project Outlay": initial_project_outlay,
        "Initial Loan Draw": initial_loan_draw,
        "Initial Equity Investment": initial_equity,
        "Cumulative FCF": _cumulative_total(raw_free_cash_flow),
        "Cumulative Equity CF": _cumulative_total(equity_cash_flow),
        "Final Month Revenue": _final_value(financials.income_monthly["Revenue"]),
        "Final Month EBITDA": _final_value(financials.income_monthly["EBITDA"]),
        "Final Month Equity CF": _final_value(equity_cash_flow),
        "Payback Period (months)": payback_months,
        "Payback Month": payback_label,
    }
    return metrics


def compute_break_even(revenue: RevenueOutput, cost_outputs: Dict[str, CostOutput]) -> pd.DataFrame:
    revenue_series = revenue.monthly["Total Revenue"]
    cost_series = sum(output.monthly.sum(axis=1) for output in cost_outputs.values())
    margin = revenue_series - cost_series
    cumulative = margin.cumsum()
    break_even_month = cumulative[cumulative >= 0].index.min()
    return pd.DataFrame(
        {
            "Monthly Margin": margin,
            "Cumulative Margin": cumulative,
            "Break-even Month": break_even_month,
        }
    )


def compute_payback(
    financials: FinancialStatements,
    revenue: RevenueOutput | None = None,
    *,
    initial_project_outlay: float | None = None,
) -> pd.DataFrame:
    (
        derived_outlay,
        _,
        adjusted_free_cash_flow,
        _,
    ) = _derive_initial_investment_components(financials, revenue)
    total_outlay = derived_outlay if initial_project_outlay is None else initial_project_outlay
    cumulative = adjusted_free_cash_flow.cumsum() - total_outlay
    payback_month = cumulative[cumulative >= 0].index.min() if not cumulative.empty else None
    return pd.DataFrame({"Cumulative FCF": cumulative, "Payback Month": payback_month})
