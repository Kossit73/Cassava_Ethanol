"""Cycle-driven farming, sourcing, processing, and commercialization engine.

The legacy model accepted a monthly cassava-production curve and applied two
fixed conversion factors. This module instead starts from annual crop cycles,
keeps the operational calculations monthly, and reconciles the full physical
and financial chain from field or supplier to commercial sale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


FINAL_PRODUCT_COLUMNS: dict[str, str] = {
    "FUEL ETHANOL": "Ethanol litres",
    "HQCF": "HQCF ton",
    "GARRI": "Garri ton",
    "INDUSTRIAL STARCH": "Industrial Starch ton",
    "DEXTRIN": "Dextrin ton",
    "GLUCOSE SYRUP": "Glucose Syrup ton",
    "SORBITOL": "Sorbitol ton",
    "ANIMAL FEED": "Animal Feed ton",
}

INTERMEDIATE_STREAMS = {"STARCH POOL", "GLUCOSE SYRUP POOL", "RESIDUE POOL"}


@dataclass
class IntegratedCycleOutput:
    """Auditable output package for the integrated Cassava value chain."""

    cycle_summary: pd.DataFrame
    monthly_physical: pd.DataFrame
    annual_physical: pd.DataFrame
    farm_monthly: pd.DataFrame
    farm_annual: pd.DataFrame
    farm_income_monthly: pd.DataFrame
    farm_income_annual: pd.DataFrame
    farm_cashflow_monthly: pd.DataFrame
    farm_cashflow_annual: pd.DataFrame
    farm_balance_monthly: pd.DataFrame
    farm_balance_annual: pd.DataFrame
    procurement_monthly: pd.DataFrame
    procurement_annual: pd.DataFrame
    processing_ledger: pd.DataFrame
    product_monthly: pd.DataFrame
    product_annual: pd.DataFrame
    commercialization_ledger: pd.DataFrame
    commercialization_monthly: pd.DataFrame
    commercialization_annual: pd.DataFrame
    revenue_monthly: pd.DataFrame
    revenue_annual: pd.DataFrame
    direct_cost_monthly: pd.DataFrame
    direct_cost_annual: pd.DataFrame
    working_capital_monthly: pd.DataFrame
    working_capital_annual: pd.DataFrame
    segment_monthly: pd.DataFrame
    segment_annual: pd.DataFrame
    eliminations_monthly: pd.DataFrame
    eliminations_annual: pd.DataFrame
    mass_balance_monthly: pd.DataFrame
    mass_balance_annual: pd.DataFrame
    validations: pd.DataFrame
    metrics: dict[str, float]


def _annual_sum(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=[])
    annual = frame.groupby(frame.index.year).sum(numeric_only=True)
    annual.index.name = "Year"
    return annual


def _annual_last(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=[])
    annual = frame.groupby(frame.index.year).last()
    annual.index.name = "Year"
    return annual


def _num(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if np.isfinite(number) else float(default)


def _pct(value: object, default: float = 0.0) -> float:
    number = _num(value, default)
    if abs(number) > 1.0:
        number /= 100.0
    return float(number)


def _normalise_name(value: object) -> str:
    return " ".join(str(value or "").strip().upper().split())


def _table_frame(page: object, attribute: str) -> pd.DataFrame:
    table = getattr(page, attribute, None)
    frame = getattr(table, "model_frame", pd.DataFrame())
    return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()


def _global_lookup(page: object) -> dict[str, object]:
    frame = _table_frame(page, "global_inputs")
    if frame.empty or not {"Parameter", "Value"}.issubset(frame.columns):
        return {}
    return frame.set_index("Parameter")["Value"].to_dict()


def _month_period(value: object, fallback: pd.Period) -> pd.Period:
    try:
        return pd.Period(str(value), freq="M")
    except Exception:
        return fallback


def _month_range(start_year: int, end_year: int) -> pd.DatetimeIndex:
    return pd.date_range(
        f"{int(start_year):04d}-01-01",
        f"{int(end_year):04d}-12-01",
        freq="MS",
    )


def _product_column(product: object, unit: object = "ton") -> str:
    key = _normalise_name(product)
    if key in FINAL_PRODUCT_COLUMNS:
        return FINAL_PRODUCT_COLUMNS[key]
    suffix = "litres" if "LIT" in _normalise_name(unit) else "ton"
    return f"{str(product).strip()} {suffix}".strip()


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    if frame is None or frame.empty:
        return []
    return frame.where(pd.notna(frame), None).to_dict("records")


def _validation_frame(messages: Iterable[tuple[str, str, str]]) -> pd.DataFrame:
    rows = [
        {"Severity": severity, "Area": area, "Message": message}
        for severity, area, message in messages
    ]
    return pd.DataFrame(rows, columns=["Severity", "Area", "Message"])


def _effective_year_row(frame: pd.DataFrame, year: int) -> Mapping[str, object]:
    if frame is None or frame.empty:
        return {}
    working = frame.copy()
    working["_year"] = pd.to_numeric(working.get("Year"), errors="coerce")
    eligible = working.loc[working["_year"].notna() & (working["_year"] <= int(year))]
    if eligible.empty:
        eligible = working.loc[working["_year"].notna()]
    if eligible.empty:
        return {}
    return eligible.sort_values("_year").iloc[-1].to_dict()


def _scale_from_base(value: object, rate: object, base_year: int, year: int) -> float:
    return _num(value) * ((1.0 + _pct(rate)) ** max(0, int(year) - int(base_year)))


def _expand_cycle_plan(
    frame: pd.DataFrame,
    start_year: int,
    end_year: int,
    planning_start: pd.Period,
) -> list[dict[str, object]]:
    """Expand annual anchor rows across the forecast horizon."""

    if frame is None or frame.empty:
        return []
    working = frame.copy()
    working["_year"] = pd.to_numeric(working.get("Year"), errors="coerce")
    working = working.loc[working["_year"].notna()].copy()
    if working.empty:
        return []
    working["_year"] = working["_year"].astype(int)

    expanded: list[dict[str, object]] = []
    first_plan_year = max(int(start_year), int(planning_start.year))
    for year in range(first_plan_year, int(end_year) + 1):
        explicit = working.loc[working["_year"] == year]
        if not explicit.empty:
            source = explicit
            source_year = year
        else:
            prior_years = working.loc[working["_year"] <= year, "_year"]
            source_year = int(prior_years.max()) if not prior_years.empty else int(working["_year"].min())
            source = working.loc[working["_year"] == source_year]

        for position, source_row in enumerate(_records(source), start=1):
            row = dict(source_row)
            delta_years = year - source_year
            row["Year"] = year
            cycle_id = str(row.get("Cycle ID", "")).strip() or f"C{position}"
            row["Cycle ID"] = f"{year}-{cycle_id.split('-', 1)[-1]}"
            for month_column in ("Planting Month", "Harvest Month", "Processing Start Month"):
                fallback = pd.Period(year=source_year, month=1, freq="M")
                source_period = _month_period(row.get(month_column), fallback)
                row[month_column] = (source_period + 12 * delta_years).strftime("%Y-%m")
            if delta_years:
                annual_increment = row.get("Annual Increment %", 0.0)
                row["Cultivated Hectares"] = _scale_from_base(
                    row.get("Cultivated Hectares"), annual_increment, source_year, year
                )
                row["Cassava Processing Target ton"] = _scale_from_base(
                    row.get("Cassava Processing Target ton"), annual_increment, source_year, year
                )
            expanded.append(row)
    return expanded


def _farming_staff_schedule(page: object, dates: pd.DatetimeIndex) -> pd.Series:
    """Return permanent farm payroll using effective-month rows and increments."""

    frame = _table_frame(page, "staff_costs_monthly")
    result = pd.Series(0.0, index=dates, dtype=float)
    if frame.empty or not {"Month", "Department", "Cost"}.issubset(frame.columns):
        return result
    working = frame.loc[
        frame["Department"].astype(str).str.contains("farm", case=False, na=False)
    ].copy()
    if working.empty:
        return result
    working["_month"] = pd.to_datetime(working["Month"].astype(str), errors="coerce")
    working["_cost"] = pd.to_numeric(working["Cost"], errors="coerce").fillna(0.0)
    working["_increment"] = pd.to_numeric(
        working.get("Annual Increment %", 0.0), errors="coerce"
    ).fillna(0.0)
    working = working.dropna(subset=["_month"]).sort_values("_month")
    for department, group in working.groupby("Department", sort=False):
        del department
        rows = group.to_dict("records")
        for month in dates:
            eligible = [row for row in rows if pd.Timestamp(row["_month"]) <= month]
            if not eligible:
                continue
            row = eligible[-1]
            anchor = pd.Timestamp(row["_month"])
            elapsed_years = max(0, month.year - anchor.year - int(month.month < anchor.month))
            result.loc[month] += float(row["_cost"]) * (
                (1.0 + _pct(row["_increment"])) ** elapsed_years
            )
    return result


def _farm_capex_schedule(
    page: object,
    dates: pd.DatetimeIndex,
    scenario: str,
    hybrid_share: float,
) -> pd.DataFrame:
    """Build farm-only fixed-asset, depreciation, and maintenance schedules."""

    result = pd.DataFrame(
        0.0,
        index=dates,
        columns=[
            "Farm Capex",
            "Farm Depreciation",
            "Farm Asset Maintenance",
            "Farm Gross PPE",
            "Farm Accumulated Depreciation",
            "Farm Net Book Value",
        ],
    )
    frame = _table_frame(page, "farm_capex")
    if frame.empty:
        return result
    scenario_scale = 0.0 if scenario == "BUY_ONLY" else (hybrid_share if scenario == "HYBRID" else 1.0)
    for row in _records(frame):
        start = _month_period(row.get("Start Month"), pd.Period(dates[0], freq="M")).to_timestamp()
        cost = max(0.0, _num(row.get("Cost"))) * scenario_scale
        life_years = max(0.0, _num(row.get("Life (years)")))
        residual = float(np.clip(_pct(row.get("Residual %")), 0.0, 1.0))
        maintenance_rate = max(0.0, _pct(row.get("Annual Maintenance %")))
        if start in result.index:
            result.loc[start, "Farm Capex"] += cost
        active_mask = result.index >= start
        result.loc[active_mask, "Farm Gross PPE"] += cost
        if life_years > 0 and cost > 0:
            monthly_depreciation = (cost * (1.0 - residual)) / max(1.0, life_years * 12.0)
            end = start + pd.DateOffset(months=max(1, int(round(life_years * 12))) - 1)
            depreciation_mask = (result.index >= start) & (result.index <= end)
            result.loc[depreciation_mask, "Farm Depreciation"] += monthly_depreciation
        if cost > 0 and maintenance_rate > 0:
            result.loc[active_mask, "Farm Asset Maintenance"] += cost * maintenance_rate / 12.0
    result["Farm Accumulated Depreciation"] = result["Farm Depreciation"].cumsum()
    result["Farm Net Book Value"] = np.maximum(
        result["Farm Gross PPE"] - result["Farm Accumulated Depreciation"], 0.0
    )
    return result


def _allocate_farm_costs(
    page: object,
    dates: pd.DatetimeIndex,
    cycles: list[dict[str, object]],
    planning_year: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Allocate every farming input to its physical crop-cycle workload."""

    frame = _table_frame(page, "farm_cost_assumptions")
    monthly = pd.DataFrame(index=dates)
    cycle_costs: dict[str, float] = {str(cycle["Cycle ID"]): 0.0 for cycle in cycles}
    for row in _records(frame):
        item = str(row.get("Cost Item", "Farm Cost")).strip() or "Farm Cost"
        stage = _normalise_name(row.get("Stage"))
        basis = _normalise_name(row.get("Basis"))
        base_cost = max(0.0, _num(row.get("Unit Cost")))
        escalation = row.get("Annual Increment %", 0.0)
        column = f"Farm - {item}"
        if column not in monthly.columns:
            monthly[column] = 0.0

        for cycle in cycles:
            cycle_id = str(cycle["Cycle ID"])
            year = int(cycle["Year"])
            cost = _scale_from_base(base_cost, escalation, planning_year, year)
            hectares = max(0.0, _num(cycle.get("Effective Farm Hectares")))
            planned_hectares = max(0.0, _num(cycle.get("Cultivated Hectares")))
            farm_scale = (
                hectares / planned_hectares if planned_hectares > 0 else 0.0
            )
            delivered = max(0.0, _num(cycle.get("Farm Cassava Delivered ton")))
            planting = pd.Period(cycle["Planting Month"], freq="M").to_timestamp()
            processing_months = [
                pd.Period(value, freq="M").to_timestamp()
                for value in cycle.get("Processing Month List", [])
            ]
            cultivation_months = [
                month
                for month in dates
                if planting <= month < (processing_months[0] if processing_months else planting)
            ]

            if stage in {"LAND PREPARATION", "PLANTING", "ESTABLISHMENT"}:
                event_months = [planting]
            elif stage in {"HARVEST", "HARVESTING"}:
                event_months = processing_months
            elif stage in {"DELIVERY", "TRANSPORT"}:
                event_months = processing_months
            elif stage in {"FIXED", "ADMINISTRATION"}:
                event_months = [month for month in dates if month.year == year]
            else:
                event_months = cultivation_months
            event_months = [month for month in event_months if month in monthly.index]
            if not event_months or cost == 0:
                continue

            if basis in {"USD/HA/MONTH", "USD PER HA PER MONTH"}:
                amounts = [hectares * cost] * len(event_months)
            elif basis in {"USD/TON HARVESTED", "USD/TON", "USD PER TON"}:
                amounts = [delivered / len(event_months) * cost] * len(event_months)
            elif basis in {"USD/MONTH", "USD PER MONTH"}:
                amounts = [cost * farm_scale] * len(event_months)
            elif basis in {"USD/CYCLE", "USD PER CYCLE"}:
                amounts = [cost * farm_scale] + [0.0] * (len(event_months) - 1)
            else:
                total = hectares * cost
                amounts = [total / len(event_months)] * len(event_months)

            for month, amount in zip(event_months, amounts):
                monthly.loc[month, column] += amount
                cycle_costs[cycle_id] = cycle_costs.get(cycle_id, 0.0) + amount

    staff = _farming_staff_schedule(page, dates)
    monthly["Farm - Permanent Farm Labour"] = staff
    for year in sorted({int(cycle["Year"]) for cycle in cycles}):
        year_cycles = [cycle for cycle in cycles if int(cycle["Year"]) == year]
        year_staff = float(staff.loc[staff.index.year == year].sum())
        total_hectares = sum(_num(cycle.get("Effective Farm Hectares")) for cycle in year_cycles)
        for cycle in year_cycles:
            weight = (
                _num(cycle.get("Effective Farm Hectares")) / total_hectares
                if total_hectares > 0
                else 1.0 / max(1, len(year_cycles))
            )
            cycle_id = str(cycle["Cycle ID"])
            cycle_costs[cycle_id] = cycle_costs.get(cycle_id, 0.0) + year_staff * weight

    monthly = monthly.fillna(0.0)
    monthly["Total Farm Operating Cost"] = monthly.sum(axis=1)
    return monthly, cycle_costs


def _normalise_cycles(
    page: object,
    scenario: str,
    dates: pd.DatetimeIndex,
    planning_start: pd.Period,
    validations: list[tuple[str, str, str]],
) -> list[dict[str, object]]:
    plan = _table_frame(page, "annual_cycle_plan")
    expanded = _expand_cycle_plan(
        plan,
        int(dates[0].year),
        int(dates[-1].year),
        planning_start,
    )
    globals_lookup = _global_lookup(page)
    default_hybrid_share = float(
        np.clip(_pct(globals_lookup.get("Hybrid farm share", 0.5)), 0.0, 1.0)
    )
    cycles: list[dict[str, object]] = []
    for position, row in enumerate(expanded, start=1):
        year = int(_num(row.get("Year"), dates[0].year))
        cycle_id = str(row.get("Cycle ID", f"{year}-C{position}"))
        planting = _month_period(row.get("Planting Month"), pd.Period(f"{year}-01", freq="M"))
        cultivation_input = int(round(_num(row.get("Cultivation Months"), 9.0)))
        if not 9 <= cultivation_input <= 12:
            validations.append(
                (
                    "ERROR",
                    "Cycle",
                    f"{cycle_id}: cultivation duration must be between 9 and 12 months; 9 months was used.",
                )
            )
        cultivation_months = int(np.clip(cultivation_input, 9, 12))
        calculated_harvest = planting + cultivation_months - 1
        supplied_harvest = _month_period(row.get("Harvest Month"), calculated_harvest)
        if supplied_harvest != calculated_harvest:
            validations.append(
                (
                    "WARNING",
                    "Cycle",
                    f"{cycle_id}: harvest month was aligned to the {cultivation_months}-month crop maturity.",
                )
            )
        harvest = calculated_harvest

        processing_input = int(round(_num(row.get("Processing Months"), 3.0)))
        if processing_input != 3:
            validations.append(
                (
                    "ERROR",
                    "Cycle",
                    f"{cycle_id}: transformation must cover exactly three months; 3 months was used.",
                )
            )
        processing_month_count = 3
        processing_start = _month_period(row.get("Processing Start Month"), harvest + 1)
        if processing_start <= harvest:
            validations.append(
                (
                    "WARNING",
                    "Cycle",
                    f"{cycle_id}: processing was moved to the month immediately after crop maturity.",
                )
            )
            processing_start = harvest + 1
        processing_periods = [processing_start + offset for offset in range(processing_month_count)]

        hectares = max(0.0, _num(row.get("Cultivated Hectares")))
        yield_per_ha = max(0.0, _num(row.get("Yield t/ha")))
        field_loss = float(np.clip(_pct(row.get("Field Loss %")), 0.0, 0.95))
        harvest_loss = float(np.clip(_pct(row.get("Harvest Loss %")), 0.0, 0.95))
        target = max(0.0, _num(row.get("Cassava Processing Target ton")))
        hybrid_share = float(
            np.clip(
                _pct(row.get("Hybrid Farm Share %"), default_hybrid_share),
                0.0,
                1.0,
            )
        )
        farm_scale = 0.0 if scenario == "BUY_ONLY" else (hybrid_share if scenario == "HYBRID" else 1.0)
        effective_hectares = hectares * farm_scale
        gross_harvest = effective_hectares * yield_per_ha
        net_harvest = gross_harvest * (1.0 - field_loss) * (1.0 - harvest_loss)
        farm_delivered = min(net_harvest, target) if scenario != "BUY_ONLY" else 0.0
        purchase_required = max(0.0, target - farm_delivered) if scenario != "FARM_ONLY" else 0.0
        shortfall = max(0.0, target - farm_delivered - purchase_required)
        farm_surplus = max(0.0, net_harvest - farm_delivered)
        if shortfall > 1e-6:
            validations.append(
                (
                    "ERROR",
                    "Sourcing",
                    f"{cycle_id}: farm output is {shortfall:,.2f} tonnes below the processing target.",
                )
            )
        if farm_surplus > 1e-6:
            validations.append(
                (
                    "INFO",
                    "Farming",
                    f"{cycle_id}: {farm_surplus:,.2f} tonnes of farm harvest remain available outside the processing transfer.",
                )
            )
        if any(period.to_timestamp() not in dates for period in processing_periods):
            validations.append(
                (
                    "WARNING",
                    "Cycle",
                    f"{cycle_id}: part of the three-month processing window falls outside the projection horizon.",
                )
            )

        cycles.append(
            {
                "Year": year,
                "Cycle ID": cycle_id,
                "Planting Month": planting.strftime("%Y-%m"),
                "Harvest Month": harvest.strftime("%Y-%m"),
                "Cultivation Months": cultivation_months,
                "Processing Start Month": processing_start.strftime("%Y-%m"),
                "Processing Months": processing_month_count,
                "Processing Month List": [period.strftime("%Y-%m") for period in processing_periods],
                "Cultivated Hectares": hectares,
                "Effective Farm Hectares": effective_hectares,
                "Yield t/ha": yield_per_ha,
                "Field Loss %": field_loss,
                "Harvest Loss %": harvest_loss,
                "Gross Farm Harvest ton": gross_harvest,
                "Net Farm Harvest ton": net_harvest,
                "Farm Cassava Delivered ton": farm_delivered,
                "Farm Surplus ton": farm_surplus,
                "Purchased Cassava Required ton": purchase_required,
                "Cassava Processing Target ton": target,
                "Feedstock Shortfall ton": shortfall,
                "Hybrid Farm Share %": hybrid_share,
            }
        )
    return cycles


def _build_procurement_schedule(
    page: object,
    dates: pd.DatetimeIndex,
    cycles: list[dict[str, object]],
    validations: list[tuple[str, str, str]],
) -> pd.DataFrame:
    frame = _table_frame(page, "procurement_plan")
    monthly = pd.DataFrame(
        0.0,
        index=dates,
        columns=[
            "Purchased Cassava Gross ton",
            "Purchased Cassava Delivered ton",
            "Purchased Quality Loss ton",
            "Contracted Cassava Cost",
            "Open Market Cassava Cost",
            "Inbound Logistics Cost",
            "Total Purchased Feedstock Cost",
            "Procurement Payables",
        ],
    )
    for cycle in cycles:
        required_net = max(0.0, _num(cycle.get("Purchased Cassava Required ton")))
        if required_net <= 0:
            cycle["Purchased Cassava Gross ton"] = 0.0
            cycle["Purchased Feedstock Cost"] = 0.0
            continue
        year = int(cycle["Year"])
        row = dict(_effective_year_row(frame, year))
        base_year = int(_num(row.get("_year"), year))
        escalation = row.get("Annual Price Increment %", 0.0)
        contracted_share = float(np.clip(_pct(row.get("Contracted Share %"), 0.6), 0.0, 1.0))
        contract_price = _scale_from_base(
            row.get("Contract Price USD/t", 70.0), escalation, base_year, year
        )
        market_price = _scale_from_base(
            row.get("Open Market Price USD/t", 80.0), escalation, base_year, year
        )
        logistics = _scale_from_base(
            row.get("Inbound Logistics USD/t", 8.0), escalation, base_year, year
        )
        quality_loss = float(np.clip(_pct(row.get("Quality Loss %"), 0.03), 0.0, 0.95))
        payable_days = max(0.0, _num(row.get("Payable Days"), 30.0))
        gross_required = required_net / max(1e-9, 1.0 - quality_loss)
        periods = [
            pd.Period(value, freq="M").to_timestamp()
            for value in cycle.get("Processing Month List", [])
        ]
        active_periods = [period for period in periods if period in monthly.index]
        if not active_periods:
            validations.append(
                (
                    "ERROR",
                    "Procurement",
                    f"{cycle['Cycle ID']}: purchased cassava cannot be scheduled outside the projection horizon.",
                )
            )
            continue
        gross_month = gross_required / len(active_periods)
        net_month = required_net / len(active_periods)
        contract_cost_month = gross_month * contracted_share * contract_price
        market_cost_month = gross_month * (1.0 - contracted_share) * market_price
        logistics_month = gross_month * logistics
        total_month = contract_cost_month + market_cost_month + logistics_month
        for period in active_periods:
            monthly.loc[period, "Purchased Cassava Gross ton"] += gross_month
            monthly.loc[period, "Purchased Cassava Delivered ton"] += net_month
            monthly.loc[period, "Purchased Quality Loss ton"] += gross_month - net_month
            monthly.loc[period, "Contracted Cassava Cost"] += contract_cost_month
            monthly.loc[period, "Open Market Cassava Cost"] += market_cost_month
            monthly.loc[period, "Inbound Logistics Cost"] += logistics_month
            monthly.loc[period, "Total Purchased Feedstock Cost"] += total_month
            monthly.loc[period, "Procurement Payables"] += total_month * payable_days / 30.0
        cycle["Purchased Cassava Gross ton"] = gross_required
        cycle["Purchased Feedstock Cost"] = total_month * len(active_periods)
    return monthly


def _route_products(
    page: object,
    dates: pd.DatetimeIndex,
    physical: pd.DataFrame,
    validations: list[tuple[str, str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Route cassava through primary and derivative stages without double counting."""

    route_frame = _table_frame(page, "product_routing")
    route_rows = _records(route_frame)
    if not route_rows:
        return pd.DataFrame(), pd.DataFrame(index=dates), pd.DataFrame(index=dates), pd.DataFrame(index=dates)
    for row in route_rows:
        row["_stage"] = int(max(1, round(_num(row.get("Stage Order"), 1))))
        row["_input"] = _normalise_name(row.get("Input Stream"))
        row["_output"] = _normalise_name(row.get("Output Stream"))
        row["_allocation"] = max(0.0, _pct(row.get("Allocation %")))
        row["_base_yield"] = max(0.0, _num(row.get("Output Yield per Input")))
        reference_dry_matter = max(0.0, _pct(row.get("Reference Dry Matter %"), 1.0))
        actual_dry_matter = max(
            0.0,
            _pct(row.get("Actual Dry Matter %"), reference_dry_matter),
        )
        dry_matter_factor = (
            actual_dry_matter / reference_dry_matter
            if row["_input"] == "CASSAVA" and reference_dry_matter > 0
            else 1.0
        )
        row["_reference_dry_matter"] = reference_dry_matter
        row["_actual_dry_matter"] = actual_dry_matter
        row["_dry_matter_factor"] = dry_matter_factor
        row["_yield"] = row["_base_yield"] * dry_matter_factor
        row["_loss"] = float(np.clip(_pct(row.get("Processing Loss %")), 0.0, 0.95))
        row["_residue"] = max(0.0, _pct(row.get("Residue Yield %")))
        row["_capacity"] = max(0.0, _num(row.get("Monthly Capacity")))
        row["_cost"] = max(0.0, _num(row.get("Processing Cost per Output Unit")))

    for row in route_rows:
        output_name = str(row["_output"])
        grade = _normalise_name(row.get("Feedstock Grade"))
        age_days = max(0.0, _num(row.get("Maximum Feedstock Age Days")))
        if output_name in {"HQCF", "GARRI"} and (
            "FRESH" not in grade or age_days > 2.0
        ):
            validations.append(
                (
                    "WARNING",
                    "Fresh Food Stream",
                    f"{output_name.title()} should use fresh roots processed within two days of harvest.",
                )
            )
        if output_name == "STARCH POOL" and (
            not any(token in grade for token in ("LARGE", "OLDER"))
            or float(row["_actual_dry_matter"]) <= float(row["_reference_dry_matter"])
        ):
            validations.append(
                (
                    "WARNING",
                    "Industrial Starch Stream",
                    "The Starch Pool should use larger/older roots with dry matter above the reference grade.",
                )
            )


    for (stage, input_stream), group in pd.DataFrame(route_rows).groupby(["_stage", "_input"]):
        allocation = float(group["_allocation"].sum())
        if allocation > 1.0 + 1e-9:
            validations.append(
                (
                    "ERROR",
                    "Product Routing",
                    f"Stage {stage} allocations from {input_stream.title()} total {allocation:.1%}; they were normalized to 100%.",
                )
            )

    product_columns = sorted(set(FINAL_PRODUCT_COLUMNS.values()))
    product_monthly = pd.DataFrame(0.0, index=dates, columns=product_columns)
    processing_cost = pd.DataFrame(index=dates)
    ledger_rows: list[dict[str, object]] = []
    balance_rows: list[dict[str, object]] = []
    globals_lookup = _global_lookup(page)
    sorting_reject_rate = float(
        np.clip(_pct(globals_lookup.get("Raw cassava sorting reject %", 0.02)), 0.0, 0.50)
    )
    residue_recovery = float(
        np.clip(_pct(globals_lookup.get("Residue recovery to feed %", 0.90)), 0.0, 1.0)
    )

    stage_numbers = sorted({int(row["_stage"]) for row in route_rows})
    for month in dates:
        delivered = max(0.0, _num(physical.loc[month, "Cassava Processed ton"]))
        raw_rejects = delivered * sorting_reject_rate
        usable = delivered - raw_rejects
        streams: dict[str, float] = {
            "CASSAVA": usable,
            "RESIDUE POOL": raw_rejects * residue_recovery,
        }
        produced_by_stream: dict[str, float] = {
            "CASSAVA": usable,
            "RESIDUE POOL": raw_rejects * residue_recovery,
        }
        routed_by_stream: dict[str, float] = {}

        for stage in stage_numbers:
            stage_rows = [row for row in route_rows if int(row["_stage"]) == stage]
            input_names = list(dict.fromkeys(str(row["_input"]) for row in stage_rows))
            for input_name in input_names:
                candidates = [row for row in stage_rows if row["_input"] == input_name]
                available = max(0.0, streams.get(input_name, 0.0))
                total_allocation = sum(float(row["_allocation"]) for row in candidates)
                allocation_scale = max(1.0, total_allocation)
                consumed_total = 0.0
                for row in candidates:
                    requested_input = available * float(row["_allocation"]) / allocation_scale
                    yield_rate = float(row["_yield"])
                    loss_rate = float(row["_loss"])
                    net_yield = yield_rate * (1.0 - loss_rate)
                    capacity = float(row["_capacity"])
                    capacity_input = (
                        capacity / net_yield
                        if capacity > 0 and net_yield > 0
                        else requested_input
                    )
                    consumed = min(requested_input, capacity_input)
                    output = consumed * net_yield
                    consumed_total += consumed
                    output_name = str(row["_output"])
                    output_type = _normalise_name(row.get("Output Type", "Product"))
                    if output_type == "INTERMEDIATE" or output_name in INTERMEDIATE_STREAMS:
                        streams[output_name] = streams.get(output_name, 0.0) + output
                        produced_by_stream[output_name] = produced_by_stream.get(output_name, 0.0) + output
                    else:
                        column = _product_column(row.get("Output Stream"), row.get("Output Unit"))
                        if column not in product_monthly.columns:
                            product_monthly[column] = 0.0
                        product_monthly.loc[month, column] += output

                    residue = consumed * float(row["_residue"])
                    if residue > 0:
                        streams["RESIDUE POOL"] = streams.get("RESIDUE POOL", 0.0) + residue
                        produced_by_stream["RESIDUE POOL"] = produced_by_stream.get("RESIDUE POOL", 0.0) + residue
                    cost_column = f"Processing - {str(row.get('Output Stream', '')).strip()}"
                    if cost_column not in processing_cost.columns:
                        processing_cost[cost_column] = 0.0
                    route_cost = output * float(row["_cost"])
                    processing_cost.loc[month, cost_column] += route_cost
                    utilization = output / capacity if capacity > 0 else 0.0
                    ledger_rows.append(
                        {
                            "Month": month,
                            "Stage Order": stage,
                            "Input Stream": str(row.get("Input Stream", "")),
                            "Output Stream": str(row.get("Output Stream", "")),
                            "Feedstock Grade": row.get("Feedstock Grade", ""),
                            "Maximum Feedstock Age Days": _num(row.get("Maximum Feedstock Age Days")),
                            "Reference Dry Matter %": row["_reference_dry_matter"],
                            "Actual Dry Matter %": row["_actual_dry_matter"],
                            "Dry Matter Yield Factor": row["_dry_matter_factor"],
                            "Input Available": available,
                            "Input Consumed": consumed,
                            "Output Quantity": output,
                            "Output Unit": row.get("Output Unit", "ton"),
                            "Processing Loss": consumed * yield_rate * loss_rate,
                            "Residue Generated ton": residue,
                            "Capacity Utilization %": utilization,
                            "Processing Cost": route_cost,
                        }
                    )
                streams[input_name] = max(0.0, available - consumed_total)
                routed_by_stream[input_name] = routed_by_stream.get(input_name, 0.0) + consumed_total

        raw_unallocated = max(0.0, streams.get("CASSAVA", 0.0))
        starch_produced = produced_by_stream.get("STARCH POOL", 0.0)
        starch_routed = routed_by_stream.get("STARCH POOL", 0.0)
        starch_unallocated = max(0.0, streams.get("STARCH POOL", 0.0))
        glucose_produced = produced_by_stream.get("GLUCOSE SYRUP POOL", 0.0)
        glucose_routed = routed_by_stream.get("GLUCOSE SYRUP POOL", 0.0)
        glucose_unallocated = max(0.0, streams.get("GLUCOSE SYRUP POOL", 0.0))
        residue_produced = produced_by_stream.get("RESIDUE POOL", 0.0)
        residue_routed = routed_by_stream.get("RESIDUE POOL", 0.0)
        residue_disposal = max(0.0, streams.get("RESIDUE POOL", 0.0))
        raw_delta = delivered - raw_rejects - routed_by_stream.get("CASSAVA", 0.0) - raw_unallocated
        starch_delta = starch_produced - starch_routed - starch_unallocated
        glucose_delta = glucose_produced - glucose_routed - glucose_unallocated
        residue_delta = residue_produced - residue_routed - residue_disposal
        balance_rows.append(
            {
                "Month": month,
                "Cassava Delivered ton": delivered,
                "Raw Sorting Rejects ton": raw_rejects,
                "Cassava Routed ton": routed_by_stream.get("CASSAVA", 0.0),
                "Unallocated Cassava ton": raw_unallocated,
                "Raw Cassava Balance Delta": raw_delta,
                "Starch Pool Produced ton": starch_produced,
                "Starch Pool Routed ton": starch_routed,
                "Unallocated Starch Pool ton": starch_unallocated,
                "Starch Balance Delta": starch_delta,
                "Glucose Pool Produced ton": glucose_produced,
                "Glucose Pool Routed ton": glucose_routed,
                "Unallocated Glucose Pool ton": glucose_unallocated,
                "Glucose Balance Delta": glucose_delta,
                "Residue Pool Generated ton": residue_produced,
                "Residue Pool Routed ton": residue_routed,
                "Residue Disposal ton": residue_disposal,
                "Residue Balance Delta": residue_delta,
            }
        )

    ledger = pd.DataFrame(ledger_rows)
    mass_balance = pd.DataFrame(balance_rows).set_index("Month") if balance_rows else pd.DataFrame(index=dates)
    for column in [
        "Raw Cassava Balance Delta",
        "Starch Balance Delta",
        "Glucose Balance Delta",
        "Residue Balance Delta",
    ]:
        if column in mass_balance.columns and (mass_balance[column].abs() > 1e-6).any():
            validations.append(
                (
                    "ERROR",
                    "Mass Balance",
                    f"{column} is not reconciled for one or more months.",
                )
            )
    return ledger, product_monthly.fillna(0.0), processing_cost.fillna(0.0), mass_balance


def _commercialize_products(
    page: object,
    dates: pd.DatetimeIndex,
    cycles: list[dict[str, object]],
    product_monthly: pd.DataFrame,
    planning_year: int,
    validations: list[tuple[str, str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply demand, inventory ageing, offtake, pricing, and selling costs."""

    frame = _table_frame(page, "commercialization_plan")
    rows = _records(frame)
    if not rows:
        empty = pd.DataFrame(index=dates)
        return pd.DataFrame(), empty, pd.DataFrame({"Total Revenue": 0.0}, index=dates)

    demand_schedule: dict[tuple[pd.Timestamp, str], float] = {}
    for cycle in cycles:
        year = int(cycle["Year"])
        periods = [
            pd.Period(value, freq="M").to_timestamp()
            for value in cycle.get("Processing Month List", [])
        ]
        periods = [period for period in periods if period in dates]
        if not periods:
            continue
        for row in rows:
            product = _normalise_name(row.get("Product"))
            annual_demand = _scale_from_base(
                row.get("Annual Demand"),
                row.get("Annual Demand Growth %", 0.0),
                planning_year,
                year,
            )
            for period in periods:
                demand_schedule[(period, product)] = (
                    demand_schedule.get((period, product), 0.0)
                    + annual_demand / len(periods)
                )

    commercial_monthly = pd.DataFrame(index=dates)
    revenue_monthly = pd.DataFrame(index=dates)
    ledger_rows: list[dict[str, object]] = []
    unmet_totals: dict[str, float] = {}
    expired_totals: dict[str, float] = {}

    for row in rows:
        product_label = str(row.get("Product", "")).strip()
        product = _normalise_name(product_label)
        unit = row.get("Unit", "ton")
        production_column = _product_column(product_label, unit)
        production = (
            pd.to_numeric(product_monthly.get(production_column), errors="coerce")
            .reindex(dates)
            .fillna(0.0)
            if production_column in product_monthly.columns
            else pd.Series(0.0, index=dates)
        )
        base_price = max(0.0, _num(row.get("Base Price")))
        price_growth = _pct(row.get("Annual Price Escalation %", 0.0))
        offtake_share = float(np.clip(_pct(row.get("Offtake Share %", 0.0)), 0.0, 1.0))
        packaging_rate = max(0.0, _num(row.get("Packaging Cost per Unit")))
        distribution_rate = max(0.0, _num(row.get("Distribution Cost per Unit")))
        marketing_rate = max(0.0, _pct(row.get("Marketing Commission %", 0.0)))
        storage_loss_rate = float(np.clip(_pct(row.get("Monthly Storage Loss %", 0.0)), 0.0, 0.95))
        storage_cost_rate = max(0.0, _num(row.get("Storage Cost per Unit per Month")))
        receivable_days = max(0.0, _num(row.get("Receivable Days"), 30.0))
        max_storage_months = max(0, int(round(_num(row.get("Maximum Storage Months"), 6.0))))
        inventory_value_rate = float(
            np.clip(_pct(row.get("Inventory Valuation %", 0.6)), 0.0, 1.0)
        )
        cohorts: list[dict[str, float]] = []
        unmet_total = 0.0
        expired_total = 0.0

        sales_column = f"Sales - {product_label}"
        inventory_column = f"Ending Inventory - {product_label}"
        commercial_cost_column = f"Commercial - {product_label}"
        revenue_column = f"{product_label} revenue"
        for column in (
            sales_column,
            inventory_column,
            commercial_cost_column,
            f"Receivables - {product_label}",
            f"Inventory Value - {product_label}",
        ):
            if column not in commercial_monthly.columns:
                commercial_monthly[column] = 0.0
        revenue_monthly[revenue_column] = 0.0

        for month in dates:
            opening_inventory = sum(cohort["quantity"] for cohort in cohorts)
            storage_loss = 0.0
            expired = 0.0
            aged: list[dict[str, float]] = []
            for cohort in cohorts:
                quantity = max(0.0, cohort["quantity"] * (1.0 - storage_loss_rate))
                storage_loss += max(0.0, cohort["quantity"] - quantity)
                age = cohort["age"] + 1.0
                if age > max_storage_months:
                    expired += quantity
                elif quantity > 1e-12:
                    aged.append({"age": age, "quantity": quantity})
            produced = max(0.0, float(production.loc[month]))
            if produced > 0:
                aged.append({"age": 0.0, "quantity": produced})
            cohorts = aged
            available = sum(cohort["quantity"] for cohort in cohorts)
            demand = max(0.0, demand_schedule.get((month, product), 0.0))
            sales = min(available, demand)
            remaining_to_sell = sales
            next_cohorts: list[dict[str, float]] = []
            for cohort in sorted(cohorts, key=lambda item: item["age"], reverse=True):
                sold = min(cohort["quantity"], remaining_to_sell)
                remaining = cohort["quantity"] - sold
                remaining_to_sell -= sold
                if remaining > 1e-12:
                    next_cohorts.append({"age": cohort["age"], "quantity": remaining})
            cohorts = next_cohorts
            ending_inventory = sum(cohort["quantity"] for cohort in cohorts)
            unmet = max(0.0, demand - sales)
            price = base_price * ((1.0 + price_growth) ** max(0, month.year - planning_year))
            revenue = sales * price
            packaging = sales * packaging_rate
            distribution = sales * distribution_rate
            marketing = revenue * marketing_rate
            storage = ending_inventory * storage_cost_rate
            commercial_cost = packaging + distribution + marketing + storage
            receivables = revenue * receivable_days / 30.0
            inventory_value = ending_inventory * price * inventory_value_rate
            contracted_sales = sales * offtake_share

            commercial_monthly.loc[month, sales_column] = sales
            commercial_monthly.loc[month, inventory_column] = ending_inventory
            commercial_monthly.loc[month, commercial_cost_column] = commercial_cost
            commercial_monthly.loc[month, f"Receivables - {product_label}"] = receivables
            commercial_monthly.loc[month, f"Inventory Value - {product_label}"] = inventory_value
            revenue_monthly.loc[month, revenue_column] = revenue
            ledger_rows.append(
                {
                    "Month": month,
                    "Product": product_label,
                    "Unit": unit,
                    "Opening Inventory": opening_inventory,
                    "Production": produced,
                    "Storage Loss": storage_loss,
                    "Expired Inventory": expired,
                    "Available for Sale": available,
                    "Demand": demand,
                    "Sales Volume": sales,
                    "Unmet Demand": unmet,
                    "Ending Inventory": ending_inventory,
                    "Contracted/Offtake Sales": contracted_sales,
                    "Spot Sales": sales - contracted_sales,
                    "Unit Price": price,
                    "Gross Revenue": revenue,
                    "Packaging Cost": packaging,
                    "Distribution Cost": distribution,
                    "Marketing Commission": marketing,
                    "Storage Cost": storage,
                    "Commercialization Cost": commercial_cost,
                    "Receivables": receivables,
                    "Inventory Value": inventory_value,
                }
            )
            unmet_total += unmet
            expired_total += expired
        unmet_totals[product_label] = unmet_total
        expired_totals[product_label] = expired_total

    commercial_cost_columns = [
        column for column in commercial_monthly.columns if column.startswith("Commercial - ")
    ]
    receivable_columns = [
        column for column in commercial_monthly.columns if column.startswith("Receivables - ")
    ]
    inventory_value_columns = [
        column for column in commercial_monthly.columns if column.startswith("Inventory Value - ")
    ]
    commercial_monthly["Total Commercialization Cost"] = (
        commercial_monthly[commercial_cost_columns].sum(axis=1)
        if commercial_cost_columns
        else 0.0
    )
    commercial_monthly["Commercial Receivables"] = (
        commercial_monthly[receivable_columns].sum(axis=1)
        if receivable_columns
        else 0.0
    )
    commercial_monthly["Finished Goods Inventory Value"] = (
        commercial_monthly[inventory_value_columns].sum(axis=1)
        if inventory_value_columns
        else 0.0
    )
    revenue_monthly["Total Revenue"] = revenue_monthly.sum(axis=1)

    for product, amount in unmet_totals.items():
        if amount > 1e-6:
            validations.append(
                (
                    "WARNING",
                    "Commercialization",
                    f"{product}: modeled production leaves {amount:,.2f} units of demand unmet.",
                )
            )
    for product, amount in expired_totals.items():
        if amount > 1e-6:
            validations.append(
                (
                    "WARNING",
                    "Commercialization",
                    f"{product}: {amount:,.2f} units expire after the maximum storage period.",
                )
            )
    ledger = pd.DataFrame(ledger_rows)
    return ledger, commercial_monthly.fillna(0.0), revenue_monthly.fillna(0.0)


def _farm_working_capital(
    farm_cost: pd.Series,
    farm_revenue: pd.Series,
    farm_deliveries: pd.Series,
    globals_lookup: Mapping[str, object],
) -> pd.DataFrame:
    receivable_days = max(0.0, _num(globals_lookup.get("Farm transfer receivable days", 30.0)))
    payable_days = max(0.0, _num(globals_lookup.get("Farm payable days", 30.0)))
    receivables = farm_revenue * receivable_days / 30.0
    payables = farm_cost * payable_days / 30.0
    wip_values: list[float] = []
    wip = 0.0
    for month in farm_cost.index:
        wip += max(0.0, float(farm_cost.loc[month]))
        if float(farm_deliveries.loc[month]) > 0:
            wip = 0.0
        wip_values.append(wip)
    wip_series = pd.Series(wip_values, index=farm_cost.index)
    return pd.DataFrame(
        {
            "Farm Receivables": receivables,
            "Biological Inventory/WIP": wip_series,
            "Farm Payables": payables,
            "Farm Net Working Capital": receivables + wip_series - payables,
        }
    )


def build_integrated_cycle(
    page: object,
    scenario: str,
    start_year: int,
    end_year: int,
    *,
    planning_start: pd.Timestamp | str | pd.Period | None = None,
) -> IntegratedCycleOutput:
    """Build the complete annual-cycle/monthly-ledger Cassava value chain."""

    scenario_name = str(scenario or "FARM_ONLY").upper()
    if scenario_name not in {"FARM_ONLY", "BUY_ONLY", "HYBRID"}:
        raise ValueError(f"Unsupported sourcing scenario: {scenario_name}")
    dates = _month_range(start_year, end_year)
    if isinstance(planning_start, pd.Period):
        planning_period = planning_start.asfreq("M")
    elif planning_start is None:
        planning_period = pd.Period(f"{start_year:04d}-01", freq="M")
    else:
        planning_period = pd.Period(pd.Timestamp(planning_start), freq="M")
    messages: list[tuple[str, str, str]] = []
    globals_lookup = _global_lookup(page)

    cycles = _normalise_cycles(
        page,
        scenario_name,
        dates,
        planning_period,
        messages,
    )
    if not cycles:
        messages.append(("ERROR", "Cycle", "No annual crop-cycle rows are available."))
    hybrid_share = float(
        np.clip(
            np.mean([_num(cycle.get("Hybrid Farm Share %"), 0.5) for cycle in cycles])
            if cycles
            else _pct(globals_lookup.get("Hybrid farm share", 0.5)),
            0.0,
            1.0,
        )
    )

    farm_monthly, cycle_operating_cost = _allocate_farm_costs(
        page,
        dates,
        cycles,
        int(planning_period.year),
    )
    farm_capex = _farm_capex_schedule(page, dates, scenario_name, hybrid_share)
    farm_monthly["Farm - Asset Maintenance"] = farm_capex["Farm Asset Maintenance"]
    operating_columns = [
        column
        for column in farm_monthly.columns
        if column.startswith("Farm - ")
    ]
    farm_monthly["Total Farm Operating Cost"] = (
        farm_monthly[operating_columns].sum(axis=1) if operating_columns else 0.0
    )

    markup = max(0.0, _pct(globals_lookup.get("Farm transfer markup %", 0.08)))
    for year in sorted({int(cycle["Year"]) for cycle in cycles}):
        year_cycles = [cycle for cycle in cycles if int(cycle["Year"]) == year]
        total_hectares = sum(_num(cycle.get("Effective Farm Hectares")) for cycle in year_cycles)
        year_depreciation = float(
            farm_capex.loc[farm_capex.index.year == year, "Farm Depreciation"].sum()
        )
        year_maintenance = float(
            farm_capex.loc[farm_capex.index.year == year, "Farm Asset Maintenance"].sum()
        )
        for cycle in year_cycles:
            weight = (
                _num(cycle.get("Effective Farm Hectares")) / total_hectares
                if total_hectares > 0
                else 1.0 / max(1, len(year_cycles))
            )
            cycle_id = str(cycle["Cycle ID"])
            cash_cost = cycle_operating_cost.get(cycle_id, 0.0) + year_maintenance * weight
            depreciation = year_depreciation * weight
            delivered = max(0.0, _num(cycle.get("Farm Cassava Delivered ton")))
            fully_loaded_cost = cash_cost + depreciation
            transfer_base = fully_loaded_cost / delivered if delivered > 0 else 0.0
            transfer_price = transfer_base * (1.0 + markup)
            transfer_revenue = delivered * transfer_price
            cycle["Farm Cash Operating Cost"] = cash_cost
            cycle["Farm Depreciation"] = depreciation
            cycle["Fully Loaded Farm Cost"] = fully_loaded_cost
            cycle["Farm Transfer Cost USD/t"] = transfer_price
            cycle["Farm Transfer Revenue"] = transfer_revenue

    procurement = _build_procurement_schedule(page, dates, cycles, messages)
    physical = pd.DataFrame(
        0.0,
        index=dates,
        columns=[
            "Cassava Feedstock Target ton",
            "Gross Farm Harvest ton",
            "Net Farm Harvest ton",
            "Farm Cassava Delivered ton",
            "Farm Surplus ton",
            "Purchased Cassava Gross ton",
            "Purchased Cassava Delivered ton",
            "Purchased Quality Loss ton",
            "Cassava Processed ton",
            "Feedstock Shortfall ton",
            "Farm Transfer Revenue",
        ],
    )
    transfer_price_weighted = pd.Series(0.0, index=dates)
    for cycle in cycles:
        periods = [
            pd.Period(value, freq="M").to_timestamp()
            for value in cycle.get("Processing Month List", [])
        ]
        periods = [period for period in periods if period in physical.index]
        if not periods:
            continue
        divisor = float(len(periods))
        farm_delivery_month = _num(cycle.get("Farm Cassava Delivered ton")) / divisor
        target_month = _num(cycle.get("Cassava Processing Target ton")) / divisor
        gross_harvest_month = _num(cycle.get("Gross Farm Harvest ton")) / divisor
        net_harvest_month = _num(cycle.get("Net Farm Harvest ton")) / divisor
        surplus_month = _num(cycle.get("Farm Surplus ton")) / divisor
        shortfall_month = _num(cycle.get("Feedstock Shortfall ton")) / divisor
        transfer_revenue_month = _num(cycle.get("Farm Transfer Revenue")) / divisor
        transfer_price = _num(cycle.get("Farm Transfer Cost USD/t"))
        for period in periods:
            physical.loc[period, "Cassava Feedstock Target ton"] += target_month
            physical.loc[period, "Gross Farm Harvest ton"] += gross_harvest_month
            physical.loc[period, "Net Farm Harvest ton"] += net_harvest_month
            physical.loc[period, "Farm Cassava Delivered ton"] += farm_delivery_month
            physical.loc[period, "Farm Surplus ton"] += surplus_month
            physical.loc[period, "Feedstock Shortfall ton"] += shortfall_month
            physical.loc[period, "Farm Transfer Revenue"] += transfer_revenue_month
            transfer_price_weighted.loc[period] += farm_delivery_month * transfer_price

    for column in (
        "Purchased Cassava Gross ton",
        "Purchased Cassava Delivered ton",
        "Purchased Quality Loss ton",
    ):
        physical[column] = procurement[column].reindex(dates).fillna(0.0)
    physical["Cassava Processed ton"] = (
        physical["Farm Cassava Delivered ton"]
        + physical["Purchased Cassava Delivered ton"]
    )
    physical["Feedstock Shortfall ton"] = np.maximum(
        physical["Cassava Feedstock Target ton"] - physical["Cassava Processed ton"],
        physical["Feedstock Shortfall ton"],
    )
    physical["Farm Transfer Cost USD/t"] = np.divide(
        transfer_price_weighted,
        physical["Farm Cassava Delivered ton"],
        out=np.zeros(len(physical)),
        where=physical["Farm Cassava Delivered ton"].to_numpy() > 0,
    )

    route_frame = _table_frame(page, "product_routing")
    route_outputs = {
        _normalise_name(value)
        for value in route_frame.get("Output Stream", pd.Series(dtype=object)).tolist()
    }
    for required in FINAL_PRODUCT_COLUMNS:
        if required not in route_outputs:
            messages.append(
                (
                    "ERROR",
                    "Product Routing",
                    f"The required product route for {required.title()} is missing.",
                )
            )
    processing_ledger, product_monthly, processing_cost, mass_balance = _route_products(
        page,
        dates,
        physical,
        messages,
    )
    commercialization_frame = _table_frame(page, "commercialization_plan")
    commercialization_products = {
        _normalise_name(value)
        for value in commercialization_frame.get("Product", pd.Series(dtype=object)).tolist()
    }
    for required in FINAL_PRODUCT_COLUMNS:
        if required not in commercialization_products:
            messages.append(
                (
                    "ERROR",
                    "Commercialization",
                    f"Commercial assumptions for {required.title()} are missing.",
                )
            )
    commercial_ledger, commercial_monthly, revenue_monthly = _commercialize_products(
        page,
        dates,
        cycles,
        product_monthly,
        int(planning_period.year),
        messages,
    )

    farm_operating_cost = farm_monthly["Total Farm Operating Cost"].reindex(dates).fillna(0.0)
    farm_revenue = physical["Farm Transfer Revenue"].reindex(dates).fillna(0.0)
    farm_depreciation = farm_capex["Farm Depreciation"].reindex(dates).fillna(0.0)
    farm_deliveries = physical["Farm Cassava Delivered ton"].reindex(dates).fillna(0.0)
    farm_income = pd.DataFrame(
        {
            "Farm Transfer Revenue": farm_revenue,
            "Farm Operating Cost": farm_operating_cost,
            "Farm EBITDA": farm_revenue - farm_operating_cost,
            "Farm Depreciation": farm_depreciation,
            "Farm EBIT": farm_revenue - farm_operating_cost - farm_depreciation,
        }
    )
    farm_wc = _farm_working_capital(
        farm_operating_cost,
        farm_revenue,
        farm_deliveries,
        globals_lookup,
    )
    farm_change_wc = farm_wc["Farm Net Working Capital"].diff().fillna(
        farm_wc["Farm Net Working Capital"]
    )
    farm_cashflow = pd.DataFrame(
        {
            "Farm Operating Cash Flow": farm_revenue - farm_operating_cost - farm_change_wc,
            "Farm Capex": -farm_capex["Farm Capex"],
        }
    )
    farm_cashflow["Farm Free Cash Flow"] = farm_cashflow.sum(axis=1)
    farm_cash_balance = farm_cashflow["Farm Free Cash Flow"].cumsum()
    farm_balance = pd.DataFrame(
        {
            "Farm Receivables": farm_wc["Farm Receivables"],
            "Biological Inventory/WIP": farm_wc["Biological Inventory/WIP"],
            "Farm Net Book Value": farm_capex["Farm Net Book Value"],
            "Farm Cash / (Funding Requirement)": farm_cash_balance,
            "Farm Payables": farm_wc["Farm Payables"],
        }
    )
    farm_balance["Farm Net Assets"] = (
        farm_balance["Farm Receivables"]
        + farm_balance["Biological Inventory/WIP"]
        + farm_balance["Farm Net Book Value"]
        + farm_balance["Farm Cash / (Funding Requirement)"]
        - farm_balance["Farm Payables"]
    )
    farm_monthly = farm_monthly.join(
        physical[
            [
                "Gross Farm Harvest ton",
                "Net Farm Harvest ton",
                "Farm Cassava Delivered ton",
                "Farm Surplus ton",
                "Farm Transfer Cost USD/t",
                "Farm Transfer Revenue",
            ]
        ],
        how="left",
    ).join(
        farm_capex[
            [
                "Farm Capex",
                "Farm Depreciation",
                "Farm Gross PPE",
                "Farm Net Book Value",
            ]
        ],
        how="left",
    )

    direct_cost = pd.DataFrame(index=dates)
    for column in operating_columns:
        direct_cost[column] = farm_monthly[column]
    direct_cost["Farm - Asset Maintenance"] = farm_capex["Farm Asset Maintenance"]
    direct_cost["Procurement - Contracted Cassava"] = procurement["Contracted Cassava Cost"]
    direct_cost["Procurement - Open Market Cassava"] = procurement["Open Market Cassava Cost"]
    direct_cost["Procurement - Inbound Logistics"] = procurement["Inbound Logistics Cost"]
    for column in processing_cost.columns:
        direct_cost[column] = processing_cost[column]
    for column in [
        item for item in commercial_monthly.columns if item.startswith("Commercial - ")
    ]:
        direct_cost[column] = commercial_monthly[column]
    direct_cost = direct_cost.fillna(0.0)

    purchased_cost = procurement["Total Purchased Feedstock Cost"].reindex(dates).fillna(0.0)
    processing_total = processing_cost.sum(axis=1).reindex(dates).fillna(0.0)
    commercial_total = commercial_monthly["Total Commercialization Cost"].reindex(dates).fillna(0.0)
    external_revenue = revenue_monthly["Total Revenue"].reindex(dates).fillna(0.0)
    transfer_expense = farm_revenue.copy()
    segment = pd.DataFrame(
        {
            "FarmCo Revenue": farm_revenue,
            "FarmCo Operating Cost": farm_operating_cost,
            "FarmCo EBITDA": farm_revenue - farm_operating_cost,
            "ProcessingCo External Revenue": external_revenue,
            "ProcessingCo Farm Transfer Cost": transfer_expense,
            "ProcessingCo Purchased Feedstock Cost": purchased_cost,
            "ProcessingCo Conversion Cost": processing_total,
            "ProcessingCo Commercialization Cost": commercial_total,
        }
    )
    segment["ProcessingCo EBITDA"] = (
        segment["ProcessingCo External Revenue"]
        - segment["ProcessingCo Farm Transfer Cost"]
        - segment["ProcessingCo Purchased Feedstock Cost"]
        - segment["ProcessingCo Conversion Cost"]
        - segment["ProcessingCo Commercialization Cost"]
    )
    segment["Consolidated Revenue"] = external_revenue
    segment["Consolidated Direct Cost"] = direct_cost.sum(axis=1)
    segment["Consolidated EBITDA Before Shared Opex"] = (
        segment["Consolidated Revenue"] - segment["Consolidated Direct Cost"]
    )
    eliminations = pd.DataFrame(
        {
            "Eliminate FarmCo Transfer Revenue": -farm_revenue,
            "Eliminate ProcessingCo Transfer Cost": farm_revenue,
        }
    )
    eliminations["Net Intercompany Elimination"] = eliminations.sum(axis=1)

    working_capital = pd.DataFrame(
        {
            "Receivables": commercial_monthly["Commercial Receivables"],
            "Inventory": commercial_monthly["Finished Goods Inventory Value"],
            "Prepaid Expenses": 0.0,
            "Other Assets": 0.0,
            "Payables": procurement["Procurement Payables"] + farm_wc["Farm Payables"],
            "Other Payables": 0.0,
        },
        index=dates,
    )
    working_capital["Net Working Capital"] = (
        working_capital["Receivables"]
        + working_capital["Inventory"]
        + working_capital["Prepaid Expenses"]
        + working_capital["Other Assets"]
        - working_capital["Payables"]
        - working_capital["Other Payables"]
    )

    monthly_physical = physical.join(product_monthly, how="left").fillna(0.0)
    monthly_physical["Cassava ton"] = monthly_physical["Cassava Processed ton"]
    annual_physical = _annual_sum(monthly_physical)
    cycle_summary = pd.DataFrame(cycles)
    unallocated_raw = float(
        mass_balance.get("Unallocated Cassava ton", pd.Series(dtype=float)).sum()
    )
    if unallocated_raw > 1e-6:
        messages.append(
            (
                "WARNING",
                "Mass Balance",
                f"{unallocated_raw:,.2f} tonnes of usable cassava are not allocated to a primary product.",
            )
        )
    residue_disposal = float(
        mass_balance.get("Residue Disposal ton", pd.Series(dtype=float)).sum()
    )
    if residue_disposal > 1e-6:
        messages.append(
            (
                "INFO",
                "Circular Feed",
                f"{residue_disposal:,.2f} tonnes of residue remain after animal-feed recovery.",
            )
        )

    metrics = {
        "Integrated Cycle Model Enabled": 1.0,
        "Cassava Processing Target (ton)": float(
            monthly_physical["Cassava Feedstock Target ton"].sum()
        ),
        "Cassava Processed (ton)": float(monthly_physical["Cassava Processed ton"].sum()),
        "Farm Cassava Delivered (ton)": float(
            monthly_physical["Farm Cassava Delivered ton"].sum()
        ),
        "Purchased Cassava Delivered (ton)": float(
            monthly_physical["Purchased Cassava Delivered ton"].sum()
        ),
        "Feedstock Shortfall (ton)": float(
            monthly_physical["Feedstock Shortfall ton"].sum()
        ),
        "Farm Transfer Cost Per Ton": float(
            np.divide(
                farm_revenue.sum(),
                farm_deliveries.sum(),
            )
            if farm_deliveries.sum() > 0
            else 0.0
        ),
        "FarmCo EBITDA": float(farm_income["Farm EBITDA"].sum()),
        "ProcessingCo EBITDA": float(segment["ProcessingCo EBITDA"].sum()),
        "Intercompany Eliminations": float(farm_revenue.sum()),
        "Product Revenue": float(external_revenue.sum()),
        "Mass Balance Passed": float(
            1.0
            if mass_balance.empty
            or all(
                (mass_balance[column].abs() <= 1e-6).all()
                for column in (
                    "Raw Cassava Balance Delta",
                    "Starch Balance Delta",
                    "Glucose Balance Delta",
                    "Residue Balance Delta",
                )
                if column in mass_balance.columns
            )
            else 0.0
        ),
        "Cycle Validation Errors": float(
            sum(1 for severity, _, _ in messages if severity == "ERROR")
        ),
    }
    for product, column in FINAL_PRODUCT_COLUMNS.items():
        metrics[f"{product.title()} Production"] = float(
            product_monthly.get(column, pd.Series(dtype=float)).sum()
        )

    return IntegratedCycleOutput(
        cycle_summary=cycle_summary,
        monthly_physical=monthly_physical,
        annual_physical=annual_physical,
        farm_monthly=farm_monthly.fillna(0.0),
        farm_annual=_annual_sum(farm_monthly.fillna(0.0)),
        farm_income_monthly=farm_income,
        farm_income_annual=_annual_sum(farm_income),
        farm_cashflow_monthly=farm_cashflow,
        farm_cashflow_annual=_annual_sum(farm_cashflow),
        farm_balance_monthly=farm_balance,
        farm_balance_annual=_annual_last(farm_balance),
        procurement_monthly=procurement,
        procurement_annual=_annual_sum(procurement),
        processing_ledger=processing_ledger,
        product_monthly=product_monthly,
        product_annual=_annual_sum(product_monthly),
        commercialization_ledger=commercial_ledger,
        commercialization_monthly=commercial_monthly,
        commercialization_annual=_annual_sum(commercial_monthly),
        revenue_monthly=revenue_monthly,
        revenue_annual=_annual_sum(revenue_monthly),
        direct_cost_monthly=direct_cost,
        direct_cost_annual=_annual_sum(direct_cost),
        working_capital_monthly=working_capital,
        working_capital_annual=_annual_last(working_capital),
        segment_monthly=segment,
        segment_annual=_annual_sum(segment),
        eliminations_monthly=eliminations,
        eliminations_annual=_annual_sum(eliminations),
        mass_balance_monthly=mass_balance,
        mass_balance_annual=_annual_sum(mass_balance),
        validations=_validation_frame(messages),
        metrics=metrics,
    )
