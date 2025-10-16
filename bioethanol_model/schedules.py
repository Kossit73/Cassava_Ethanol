from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from .utils import annual_periods, irr, npv, year_month_range


@dataclass
class DepreciationOutput:
    monthly: pd.DataFrame
    annual: pd.DataFrame
    summary: pd.DataFrame
    capex: pd.Series


def compute_depreciation_schedule(initial_investment: pd.DataFrame, start_year: int, end_year: int) -> DepreciationOutput:
    months = year_month_range(start_year, end_year)
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


def compute_production_tables(production_monthly: pd.DataFrame, start_year: int, end_year: int) -> ProductionOutput:
    monthly = production_monthly.copy()
    if monthly.empty:
        empty = pd.DataFrame(columns=["Cassava ton", "Ethanol litres", "Animal Feed ton"])
        empty.index = pd.Index([], name="Month")
        return ProductionOutput(empty, empty)

    monthly["Month"] = pd.to_datetime(monthly["Month"].astype(str)).dt.to_period("M").dt.to_timestamp()
    monthly = monthly.sort_values("Month").reset_index(drop=True)

    growth_col = next((c for c in monthly.columns if "growth" in c.lower()), None)
    growth_series = pd.Series(dtype=float)
    if growth_col and growth_col in monthly.columns:
        growth_series = pd.to_numeric(monthly[growth_col], errors="coerce")

    monthly = monthly.set_index("Month")
    if growth_col and growth_col in monthly.columns:
        monthly = monthly.drop(columns=[growth_col])

    numeric_cols = [c for c in monthly.columns if c != growth_col]
    months = year_month_range(start_year, end_year)
    compound_monthly = pd.DataFrame(index=months)

    # Pre-compute dictionaries for quick lookup.
    growth_lookup: Dict[pd.Timestamp, float] = {}
    if not growth_series.empty:
        growth_lookup = {
            idx if isinstance(idx, pd.Timestamp) else pd.Timestamp(idx): float(val)
            for idx, val in growth_series.dropna().items()
        }

    for col in numeric_cols:
        base_series = pd.to_numeric(monthly.get(col, pd.Series(dtype=float)), errors="coerce")
        base_lookup = {
            idx if isinstance(idx, pd.Timestamp) else pd.Timestamp(idx): float(val)
            for idx, val in base_series.dropna().items()
        }

        values = []
        prev_value = None
        current_growth = 0.0
        for month in months:
            if month in growth_lookup:
                new_growth = growth_lookup[month]
                if pd.notna(new_growth):
                    current_growth = float(new_growth)
            month_value = base_lookup.get(month)
            if month_value is not None:
                value = month_value
            elif prev_value is not None:
                monthly_factor = 1.0 + current_growth / 12.0
                value = prev_value * monthly_factor
            else:
                value = 0.0

            values.append(value)
            if np.isfinite(value):
                prev_value = float(value)

        compound_monthly[col] = values

    compound_monthly = compound_monthly.sort_index()
    monthly = compound_monthly
    annual = monthly.resample("Y").sum()
    annual.index = annual.index.year
    return ProductionOutput(monthly, annual)


@dataclass
class RevenueOutput:
    monthly: pd.DataFrame
    annual: pd.DataFrame


def compute_revenue_schedule(production: ProductionOutput, revenue_inputs: pd.DataFrame, inflation_schedule: pd.DataFrame) -> RevenueOutput:
    monthly = production.monthly.copy()
    prices = {}
    for _, row in revenue_inputs.iterrows():
        product = row["Product"]
        base_price = row["Base Price"]
        escalation = row.get("Escalation", 0.0)
        prices[product] = (base_price, escalation)

    inflation = inflation_schedule.set_index("Year")["CPI"].to_dict()

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
            years_from_start = ts.year - monthly.index[0].year
            price = base_price * ((1 + escalation) ** years_from_start)
            cpi = inflation.get(ts.year, 0.0)
            price *= (1 + cpi)
            price_series.append(price)
        monthly_revenue[f"{product} revenue"] = volumes.values * np.array(price_series)
    monthly_revenue["Total Revenue"] = monthly_revenue.sum(axis=1)

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

    def _prepare(df: pd.DataFrame, value_column: str = "Amount") -> pd.DataFrame:
        copy = df.copy()
        copy["Month"] = pd.to_datetime(copy["Month"].astype(str)).dt.to_period("M").dt.to_timestamp()
        pivot = copy.pivot_table(index="Month", columns=df.columns[1], values=value_column, aggfunc="sum", fill_value=0)
        pivot = pivot.sort_index().reindex(months)
        pivot = pivot.ffill().fillna(0.0)
        return pivot

    direct = _prepare(direct_costs)
    staff = _prepare(staff_costs, value_column="Cost")
    other = _prepare(other_opex)

    outputs = {}
    for name, table in {
        "Direct Costs": direct,
        "Staff Costs": staff,
        "Other Opex": other,
    }.items():
        annual = table.resample("Y").sum()
        annual.index = annual.index.year
        outputs[name] = CostOutput(table, annual)
    return outputs


@dataclass
class LoanScheduleOutput:
    schedule: pd.DataFrame
    summary: pd.DataFrame


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
        )

    amount_columns = ["Loan Amount", "Amount", "Drawdown"]

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
        try:
            rate = float(loan.get("Interest Rate", 0.08))
        except (TypeError, ValueError):
            rate = 0.0
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
                }
            )

    schedule = pd.DataFrame(schedule_rows)
    if schedule.empty:
        summary = pd.DataFrame(columns=["Draw", "Interest", "Principal", "Payment"])
    else:
        summary = schedule.groupby("Loan").agg({"Draw": "sum", "Interest": "sum", "Principal": "sum", "Payment": "sum"})
    return LoanScheduleOutput(schedule, summary)


@dataclass
class WorkingCapitalOutput:
    monthly: pd.DataFrame
    annual: pd.DataFrame


def compute_working_capital(
    revenue: RevenueOutput,
    cost_outputs: Dict[str, CostOutput],
    ar_days: float,
    inventory_days: float,
    ap_days: float,
) -> WorkingCapitalOutput:
    monthly_revenue = revenue.monthly["Total Revenue"]
    monthly_cogs = (
        cost_outputs["Direct Costs"].monthly.sum(axis=1)
        + cost_outputs["Staff Costs"].monthly.sum(axis=1)
        + cost_outputs["Other Opex"].monthly.sum(axis=1)
    )
    days_in_month = monthly_revenue.index.days_in_month
    receivables = monthly_revenue * (ar_days / 30)
    inventory = monthly_cogs * (inventory_days / 30)
    payables = monthly_cogs * (ap_days / 30)
    wc = pd.DataFrame(
        {
            "Receivables": receivables,
            "Inventory": inventory,
            "Payables": payables,
            "Net Working Capital": receivables + inventory - payables,
        }
    )
    annual = wc.resample("Y").mean()
    annual.index = annual.index.year
    return WorkingCapitalOutput(wc, annual)


@dataclass
class FinancialStatements:
    income_monthly: pd.DataFrame
    income_annual: pd.DataFrame
    balance_monthly: pd.DataFrame
    balance_annual: pd.DataFrame
    cashflow_monthly: pd.DataFrame
    cashflow_annual: pd.DataFrame


def compute_financial_statements(
    revenue: RevenueOutput,
    depreciation: DepreciationOutput,
    cost_outputs: Dict[str, CostOutput],
    loan_schedule: LoanScheduleOutput,
    working_capital: WorkingCapitalOutput,
    tax_rate: float,
) -> FinancialStatements:
    monthly = revenue.monthly.copy()
    dep = depreciation.monthly["Total Depreciation"]
    direct = cost_outputs["Direct Costs"].monthly.sum(axis=1)
    staff = cost_outputs["Staff Costs"].monthly.sum(axis=1)
    other = cost_outputs["Other Opex"].monthly.sum(axis=1)
    interest = loan_schedule.schedule.pivot_table(index="Month", values="Interest", aggfunc="sum")
    interest = interest.reindex(monthly.index, fill_value=0.0)["Interest"]
    if "Draw" in loan_schedule.schedule.columns:
        debt_draws = (
            loan_schedule.schedule.pivot_table(index="Month", values="Draw", aggfunc="sum")
            .reindex(monthly.index, fill_value=0.0)["Draw"]
        )
    else:
        debt_draws = pd.Series(0.0, index=monthly.index)
    principal = loan_schedule.schedule.pivot_table(index="Month", values="Principal", aggfunc="sum")
    principal = principal.reindex(monthly.index, fill_value=0.0)["Principal"]
    debt_service = principal + interest
    capex = depreciation.capex.reindex(monthly.index, fill_value=0.0)

    income_monthly = pd.DataFrame(index=monthly.index)
    income_monthly["Revenue"] = monthly["Total Revenue"]
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

    wc = working_capital.monthly["Net Working Capital"]
    delta_wc = wc.diff().fillna(wc)

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
    balance_monthly["Working Capital"] = wc
    balance_monthly["Debt"] = (
        loan_schedule.schedule.pivot_table(index="Month", values="Closing Balance", aggfunc="sum")
        .reindex(monthly.index, fill_value=0.0)["Closing Balance"]
    )
    balance_monthly["Equity"] = (
        balance_monthly[["Cash", "Net PP&E", "Working Capital"]].sum(axis=1) - balance_monthly["Debt"]
    )

    balance_annual = balance_monthly.resample("Y").last()
    balance_annual.index = balance_annual.index.year

    return FinancialStatements(
        income_monthly=income_monthly,
        income_annual=income_annual,
        balance_monthly=balance_monthly,
        balance_annual=balance_annual,
        cashflow_monthly=cashflow_monthly,
        cashflow_annual=cashflow_annual,
    )


def compute_key_metrics(
    financials: FinancialStatements,
    discount_rate: float,
    investor_share: float,
    owner_share: float,
) -> Dict[str, float]:
    free_cash_flow = financials.cashflow_monthly["Free Cash Flow"].astype(float)
    equity_cash_flow = financials.cashflow_monthly["Equity Cash Flow"].astype(float)

    project_cashflows = [0.0] + free_cash_flow.tolist()
    equity_cashflows = [0.0] + equity_cash_flow.tolist()
    investor_cashflows = [0.0] + (equity_cash_flow * investor_share).tolist()
    owner_cashflows = [0.0] + (equity_cash_flow * owner_share).tolist()

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
        "Cumulative FCF": float(np.cumsum(free_cash_flow).iloc[-1]),
        "Cumulative Equity CF": float(np.cumsum(equity_cash_flow).iloc[-1]),
        "Final Month Revenue": float(financials.income_monthly["Revenue"].iloc[-1]),
        "Final Month EBITDA": float(financials.income_monthly["EBITDA"].iloc[-1]),
        "Final Month Equity CF": float(equity_cash_flow.iloc[-1]),
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


def compute_payback(cashflow_monthly: pd.DataFrame) -> pd.DataFrame:
    cumulative = cashflow_monthly["Free Cash Flow"].cumsum()
    payback_month = cumulative[cumulative >= 0].index.min()
    return pd.DataFrame({"Cumulative FCF": cumulative, "Payback Month": payback_month})
