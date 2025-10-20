from __future__ import annotations

import base64
import logging
import math
import os
import signal
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from functools import wraps
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape

try:  # pragma: no cover - lightweight shim for offline execution
    import numpy  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    from tools.numpy_stub import install_numpy_stub

    install_numpy_stub()


LOGGER = logging.getLogger(__name__)


try:  # pragma: no cover - optional dependencies
    import pandas as pd  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependencies
    pd = None  # type: ignore

try:  # pragma: no cover - optional dependencies
    import numpy as np  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependencies
    np = None  # type: ignore

try:  # pragma: no cover - optional dependencies
    import numpy_financial as npf  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependencies
    npf = None  # type: ignore

try:  # pragma: no cover - optional dependencies
    from scipy.stats import (  # type: ignore
        norm,
        lognorm,
        uniform,
        expon,
        binom,
        poisson,
        geom,
        bernoulli,
        chi2,
        gamma,
        weibull_min,
        hypergeom,
        multinomial,
        beta,
        f,
    )
except ModuleNotFoundError:  # pragma: no cover - optional dependencies
    (
        norm,
        lognorm,
        uniform,
        expon,
        binom,
        poisson,
        geom,
        bernoulli,
        chi2,
        gamma,
        weibull_min,
        hypergeom,
        multinomial,
        beta,
        f,
    ) = (None,) * 15

try:  # pragma: no cover - optional dependencies
    from scipy import stats  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependencies
    stats = None  # type: ignore

try:  # pragma: no cover - optional dependencies
    from scipy.optimize import minimize  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependencies
    minimize = None  # type: ignore

try:  # pragma: no cover - optional dependencies
    from sklearn.linear_model import LinearRegression  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependencies
    LinearRegression = None  # type: ignore

try:  # pragma: no cover - optional dependencies
    from sklearn.preprocessing import StandardScaler  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependencies
    StandardScaler = None  # type: ignore

try:  # pragma: no cover - optional dependencies
    from sklearn.neural_network import MLPRegressor  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependencies
    MLPRegressor = None  # type: ignore

try:  # pragma: no cover - optional dependencies
    from statsmodels.tsa.arima.model import ARIMA  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependencies
    ARIMA = None  # type: ignore

try:  # pragma: no cover - optional dependencies
    import networkx as nx  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependencies
    nx = None  # type: ignore

try:  # pragma: no cover - optional dependencies
    import plotly.graph_objects as go  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependencies
    go = None  # type: ignore

try:  # pragma: no cover - optional dependencies
    import matplotlib.pyplot as plt  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependencies
    plt = None  # type: ignore

try:  # pragma: no cover - optional dependencies
    from fastapi import HTTPException  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependencies
    HTTPException = None  # type: ignore

try:  # pragma: no cover - optional dependencies
    import plotly.io as pio  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependencies
    pio = None  # type: ignore

try:  # pragma: no cover - optional dependencies
    from xlsxwriter.exceptions import DuplicateWorksheetName  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependencies
    DuplicateWorksheetName = None  # type: ignore


SCENARIOS = ("FARM_ONLY", "BUY_ONLY", "HYBRID")
MONTHS_PER_YEAR = 12
ANALYSIS_HORIZON_YEARS = 10


def _projection_months(
    *, start_year: int = 2025, start_month: int = 1, years: int = ANALYSIS_HORIZON_YEARS
) -> List[str]:
    """Return ``YYYY-MM`` labels covering the full projection horizon."""

    months: List[str] = []
    year = start_year
    month = start_month
    total_months = years * MONTHS_PER_YEAR
    for _ in range(total_months):
        months.append(f"{year}-{month:02d}")
        month += 1
        if month > 12:
            month = 1
            year += 1
    return months


MONTHS = _projection_months()

_BASE_RAMP_UP_PROFILE: Sequence[float] = (
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
    1.00,
    1.00,
    1.00,
    1.00,
    1.00,
    1.00,
    1.00,
)


def _ramp_profile(months: Sequence[str]) -> List[float]:
    """Extend the first-year ramp profile across the projection horizon."""

    base = list(_BASE_RAMP_UP_PROFILE)
    length = len(months)
    if len(base) < length:
        base.extend([1.0] * (length - len(base)))
    return base[:length]


RAMP_UP_PROFILE: Sequence[float] = tuple(_ramp_profile(MONTHS))

ETHANOL_LITRES_PER_TON = 200.0
ANIMAL_FEED_TON_PER_TON = 0.275
ETHANOL_PRICE = 0.70
ANIMAL_FEED_PRICE = 120.0
FARM_PRODUCTION_COST_PER_TON = 45.0
PURCHASE_COST_PER_TON = 70.0
# Inflation and risk adjustments are compounded monthly so the operating
# schedules grow over the projection horizon instead of staying flat after the
# ramp-up completes.
ANNUAL_INFLATION_RATE = 0.05
ANNUAL_RISK_RATE = 0.02
# ``DIRECT_COST_OTHER`` represents conversion costs such as utilities and
# enzymes that move broadly with production.  When constructing the monthly
# schedules we distribute this bucket across the ramp-up profile instead of
# assuming a flat run-rate from the outset.
DIRECT_COST_OTHER = 150_000.0
OPERATIONS_STAFF_COST = 120_000.0
FARMING_STAFF_COST = 65_000.0
OTHER_OPEX_MONTHLY = 42_000.0 + 30_000.0 + 82_000.0 + 25_000.0 + 15_000.0 + 165_000.0
TAX_RATE = 0.28
LOAN_AMOUNT = 24_000_000.0
LOAN_INTEREST_RATE = 0.075
LOAN_TERM_YEARS = 10
MAINTENANCE_CAPEX_RATE = 0.03
DISCOUNT_RATE = 0.12
BASE_CASSAVA_TON_PER_MONTH = 10_000.0
ACCOUNTS_RECEIVABLE_DAYS = 45.0
OTHER_ASSETS_DAYS = 15.0
INVENTORY_DAYS = 30.0
ACCOUNTS_PAYABLE_DAYS = 30.0
OTHER_PAYABLES_DAYS = 30.0

CAPEX_ITEMS: Sequence[Tuple[str, float, float]] = (
    ("Land", 2_000_000.0, 0.0),
    ("Building", 12_000_000.0, 25.0),
    ("Plant & Equipment", 18_000_000.0, 15.0),
    ("Farm Development", 3_000_000.0, 10.0),
    ("EPC & Others", 5_000_000.0, 8.0),
)


def _first_year(values: Sequence[float]) -> List[float]:
    return list(values[:MONTHS_PER_YEAR])


def _first_year_sum(values: Sequence[float]) -> float:
    return sum(_first_year(values))


def _extend_monthly_schedule(first_year_values: Sequence[float], months: Sequence[str]) -> List[float]:
    values = list(first_year_values)
    if not values:
        return [0.0] * len(months)
    last_value = values[-1]
    while len(values) < len(months):
        values.append(last_value)
    return values[: len(months)]


@dataclass(frozen=True)
class MonthlyRunRate:
    cassava_ton_farm: float
    cassava_ton_purchase: float
    cassava_ton_total: float
    ethanol_litres: float
    animal_feed_ton: float
    ethanol_revenue: float
    animal_feed_revenue: float
    total_revenue: float
    feedstock_cost_farm: float
    feedstock_cost_purchase: float
    other_direct_cost: float
    total_direct_cost: float
    staff_cost: float
    other_opex: float
    ebitda: float
    ebitda_margin: Optional[float]


@dataclass(frozen=True)
class ScenarioData:
    scenario: str
    farm_share: float
    steady_state: MonthlyRunRate
    monthly_staff_cost: float
    monthly_other_opex: float
    annual_cassava_tons: float
    annual_ethanol_litres: float
    annual_animal_feed_ton: float
    annual_ethanol_revenue: float
    annual_coproduct_revenue: float
    annual_revenue: float
    annual_feedstock_cost_farm: float
    annual_feedstock_cost_purchase: float
    annual_direct_cost_other: float
    annual_direct_cost: float
    annual_staff_cost: float
    annual_other_opex: float
    annual_ebitda: float
    annual_depreciation: float
    operating_income: float
    interest_expense: float
    pre_tax_income: float
    taxes: float
    net_income: float
    operating_cash_flow: float
    maintenance_capex: float
    capex_total: float
    free_cash_flow: float
    annual_debt_service: float
    annual_equity_cash_flow: float
    debt_service_coverage_ratio: Optional[float]
    interest_coverage_ratio: Optional[float]
    gross_margin: Optional[float]
    ebitda_margin: Optional[float]
    net_margin: Optional[float]
    break_even_ethanol_price: Optional[float]
    payback_years: Optional[float]
    project_npv: float
    project_irr: Optional[float]
    working_capital_investment: float
    net_working_capital_end: float
    monthly_cogs_schedule: List[float]
    monthly_staff_cost_schedule: List[float]
    monthly_other_opex_schedule: List[float]
    monthly_revenue_schedule: List[float]
    monthly_ebitda_schedule: List[float]
    monthly_accounts_receivable_other_assets_schedule: List[float]
    monthly_inventory_schedule: List[float]
    monthly_accounts_payable_schedule: List[float]
    monthly_other_payables_schedule: List[float]
    net_working_capital_schedule: List[float]
    change_in_net_working_capital_schedule: List[float]
    monthly_rows: List[List[float | str]]


def _column_letter(index: int) -> str:
    if index < 1:
        raise ValueError("Excel columns are 1-indexed")
    letters: List[str] = []
    while index:
        index, remainder = divmod(index - 1, 26)
        letters.append(chr(65 + remainder))
    return "".join(reversed(letters))


def _format_number(value: float) -> float | int:
    if isinstance(value, (int,)):
        return value
    if not math.isfinite(value):
        return 0.0
    rounded = round(value, 2)
    if abs(rounded - round(rounded)) < 1e-9:
        return int(round(rounded))
    return rounded


def _build_sheet(rows: Sequence[Sequence[float | int | str | None]]) -> bytes:
    xml_rows: List[str] = []
    for row_idx, row in enumerate(rows, start=1):
        cells: List[str] = []
        for col_idx, value in enumerate(row, start=1):
            if value is None or value == "":
                continue
            ref = f"{_column_letter(col_idx)}{row_idx}"
            if isinstance(value, (int, float)):
                number = _format_number(float(value))
                cells.append(f'<c r="{ref}"><v>{number}</v></c>')
            else:
                text = escape(str(value))
                cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>')
        xml_rows.append(f'<row r="{row_idx}">{"".join(cells)}</row>')
    sheet_xml = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{''.join(xml_rows)}</sheetData>"
        "</worksheet>"
    )
    return sheet_xml.encode("utf-8")


def _spread_over_months(
    total: float,
    weights: Optional[Sequence[float]] = None,
    *,
    base_share: float = 0.0,
) -> List[float]:
    """Allocate an annual total across the modelling months.

    ``weights`` skew the distribution so the variable portion of the cost tracks
    the production ramp.  ``base_share`` designates the fraction of the annual
    total that should stay flat through the year (for fixed staffing, for
    example).  The fixed and variable components always sum to ``total``.
    """

    if weights is None:
        months = len(MONTHS)
        weights = [1.0] * months
    else:
        months = len(weights)
        if months == 0:
            return []
        weights = list(weights)

    base_share_clamped = min(max(base_share, 0.0), 1.0)
    base_total = total * base_share_clamped
    variable_total = total - base_total
    base_monthly = base_total / months if months else 0.0

    total_weight = sum(weights)
    if not math.isfinite(total_weight) or total_weight <= 0 or variable_total == 0:
        even_variable = variable_total / months if months else 0.0
        return [base_monthly + even_variable for _ in range(months)]

    return [base_monthly + variable_total * (weight / total_weight) for weight in weights]


def _compound_with_inflation_and_risk(
    values: Sequence[float],
    *,
    annual_inflation_rate: float = ANNUAL_INFLATION_RATE,
    annual_risk_rate: float = ANNUAL_RISK_RATE,
) -> List[float]:
    """Apply compounded monthly inflation and risk adjustments to ``values``."""

    if not values:
        return []

    # Guard against negative rates that could collapse the schedule.
    inflation_base = max(1.0 + annual_inflation_rate, 1e-6)
    risk_base = max(1.0 + annual_risk_rate, 1e-6)

    monthly_inflation = math.pow(inflation_base, 1.0 / 12.0) - 1.0
    monthly_risk = math.pow(risk_base, 1.0 / 12.0) - 1.0
    monthly_multiplier = (1.0 + monthly_inflation) * (1.0 + monthly_risk)
    if not math.isfinite(monthly_multiplier) or monthly_multiplier <= 0:
        monthly_multiplier = 1.0

    compounded: List[float] = []
    factor = 1.0
    for idx, value in enumerate(values):
        if idx == 0:
            factor = 1.0
        else:
            factor *= monthly_multiplier
        compounded.append(value * factor)
    return compounded


def _write_xlsx(path: Path, sheets: Sequence[Tuple[str, Sequence[Sequence[float | int | str | None]]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        sheet_entries = []
        for idx, (name, rows) in enumerate(sheets, start=1):
            entry = f"xl/worksheets/sheet{idx}.xml"
            sheet_entries.append((idx, name, entry, _build_sheet(rows)))

        overrides = "\n".join(
            f"  <Override PartName=\"/{entry}\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.work"  # noqa: E501
            "sheet+xml\"/>"
            for _, _, entry, _ in sheet_entries
        )

        content_types_xml = (
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n"
            "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">\n"
            "  <Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>\n"
            "  <Default Extension=\"xml\" ContentType=\"application/xml\"/>\n"
            "  <Override PartName=\"/xl/workbook.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml"
            ".sheet.main+xml\"/>\n"
            "  <Override PartName=\"/xl/styles.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml"
            ".styles+xml\"/>\n"
            f"{overrides}\n"
            "</Types>\n"
        )
        zf.writestr("[Content_Types].xml", content_types_xml)

        root_rels_xml = (
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n"
            "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">\n"
            "  <Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocume"
            "nt\" Target=\"xl/workbook.xml\"/>\n"
            "</Relationships>\n"
        )
        zf.writestr("_rels/.rels", root_rels_xml)

        workbook_rels = "\n".join(
            f"  <Relationship Id=\"rId{idx}\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/workshe"  # noqa: E501
            f"et\" Target=\"worksheets/sheet{idx}.xml\"/>"
            for idx, _, _, _ in sheet_entries
        )
        workbook_rels_xml = (
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n"
            "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">\n"
            f"{workbook_rels}\n"
            "</Relationships>\n"
        )
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)

        sheet_defs = "\n".join(
            f"    <sheet name=\"{escape(name)}\" sheetId=\"{idx}\" r:id=\"rId{idx}\"/>"
            for idx, name, _, _ in sheet_entries
        )
        workbook_xml = (
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n"
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">\n'
            "  <sheets>\n"
            f"{sheet_defs}\n"
            "  </sheets>\n"
            "</workbook>\n"
        )
        zf.writestr("xl/workbook.xml", workbook_xml)

        styles_xml = (
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n"
            "<styleSheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\">\n"
            "  <fonts count=\"1\"><font><sz val=\"11\"/><color theme=\"1\"/><name val=\"Calibri\"/><family val=\"2\"/></font></fonts>\n"
            "  <fills count=\"1\"><fill><patternFill patternType=\"none\"/></fill></fills>\n"
            "  <borders count=\"1\"><border><left/><right/><top/><bottom/><diagonal/></border></borders>\n"
            "  <cellStyleXfs count=\"1\"><xf numFmtId=\"0\" fontId=\"0\" fillId=\"0\" borderId=\"0\"/></cellStyleXfs>\n"
            "  <cellXfs count=\"1\"><xf numFmtId=\"0\" fontId=\"0\" fillId=\"0\" borderId=\"0\" xfId=\"0\"/></cellXfs>\n"
            "  <cellStyles count=\"1\"><cellStyle name=\"Normal\" xfId=\"0\" builtinId=\"0\"/></cellStyles>\n"
            "</styleSheet>\n"
        )
        zf.writestr("xl/styles.xml", styles_xml)

        for _, _, entry, data in sheet_entries:
            zf.writestr(entry, data)


def _net_present_value(cash_flows: Sequence[float], discount_rate: float) -> float:
    return sum(cash_flow / (1.0 + discount_rate) ** idx for idx, cash_flow in enumerate(cash_flows))


def _internal_rate_of_return(cash_flows: Sequence[float]) -> Optional[float]:
    if not cash_flows or all(value >= 0 for value in cash_flows[1:]):
        return None

    def npv(rate: float) -> float:
        return sum(value / (1.0 + rate) ** idx for idx, value in enumerate(cash_flows))

    low, high = -0.9, 1.0
    npv_low = npv(low)
    npv_high = npv(high)
    for _ in range(100):
        if npv_low * npv_high <= 0:
            break
        high += 1.0
        npv_high = npv(high)
        if high > 25.0:
            return None

    if npv_low * npv_high > 0:
        return None

    for _ in range(100):
        mid = (low + high) / 2.0
        npv_mid = npv(mid)
        if abs(npv_mid) < 1e-6:
            return mid
        if npv_low * npv_mid < 0:
            high, npv_high = mid, npv_mid
        else:
            low, npv_low = mid, npv_mid
    return (low + high) / 2.0


def _scenario_parameters(scenario: str) -> ScenarioData:
    scenario_upper = scenario.upper()
    if scenario_upper not in SCENARIOS:
        raise ValueError(f"Unsupported scenario '{scenario}'. Expected one of {SCENARIOS}.")

    farm_share_lookup = {"FARM_ONLY": 1.0, "BUY_ONLY": 0.0, "HYBRID": 0.5}
    farm_share = farm_share_lookup[scenario_upper]

    monthly_cassava_tons = [BASE_CASSAVA_TON_PER_MONTH * factor for factor in RAMP_UP_PROFILE]
    farmland_tons_schedule = [tons * farm_share for tons in monthly_cassava_tons]
    purchased_tons_schedule = [tons - farm for tons, farm in zip(monthly_cassava_tons, farmland_tons_schedule)]

    monthly_ethanol_litres = [tons * ETHANOL_LITRES_PER_TON for tons in monthly_cassava_tons]
    monthly_animal_feed_ton = [tons * ANIMAL_FEED_TON_PER_TON for tons in monthly_cassava_tons]

    base_ethanol_revenue = [litres * ETHANOL_PRICE for litres in monthly_ethanol_litres]
    base_coproduct_revenue = [feed * ANIMAL_FEED_PRICE for feed in monthly_animal_feed_ton]
    monthly_ethanol_revenue = _compound_with_inflation_and_risk(base_ethanol_revenue)
    monthly_coproduct_revenue = _compound_with_inflation_and_risk(base_coproduct_revenue)
    monthly_revenue_schedule = [eth + coproduct for eth, coproduct in zip(monthly_ethanol_revenue, monthly_coproduct_revenue)]

    base_farmland_feedstock_cost_schedule = [tons * FARM_PRODUCTION_COST_PER_TON for tons in farmland_tons_schedule]
    base_purchased_feedstock_cost_schedule = [tons * PURCHASE_COST_PER_TON for tons in purchased_tons_schedule]
    farmland_feedstock_cost_schedule = _compound_with_inflation_and_risk(base_farmland_feedstock_cost_schedule)
    purchased_feedstock_cost_schedule = _compound_with_inflation_and_risk(base_purchased_feedstock_cost_schedule)
    monthly_feedstock_cost_schedule = [
        farm + purchase for farm, purchase in zip(farmland_feedstock_cost_schedule, purchased_feedstock_cost_schedule)
    ]
    annual_other_direct_cost = DIRECT_COST_OTHER * MONTHS_PER_YEAR
    first_year_other_direct = _spread_over_months(
        annual_other_direct_cost,
        _BASE_RAMP_UP_PROFILE,
    )
    base_other_direct_schedule = _extend_monthly_schedule(first_year_other_direct, MONTHS)
    monthly_other_direct_cost_schedule = _compound_with_inflation_and_risk(base_other_direct_schedule)
    monthly_cogs_schedule = [
        feedstock + other_direct
        for feedstock, other_direct in zip(monthly_feedstock_cost_schedule, monthly_other_direct_cost_schedule)
    ]

    first_year_staff_total = (OPERATIONS_STAFF_COST + FARMING_STAFF_COST * farm_share) * MONTHS_PER_YEAR
    first_year_staff = _spread_over_months(
        first_year_staff_total,
        _BASE_RAMP_UP_PROFILE,
        base_share=0.4,
    )
    base_staff_schedule = _extend_monthly_schedule(first_year_staff, MONTHS)
    monthly_staff_cost_schedule = _compound_with_inflation_and_risk(base_staff_schedule)
    first_year_other_opex_total = OTHER_OPEX_MONTHLY * MONTHS_PER_YEAR
    first_year_other_opex = _spread_over_months(
        first_year_other_opex_total,
        _BASE_RAMP_UP_PROFILE,
        base_share=0.25,
    )
    base_other_opex_schedule = _extend_monthly_schedule(first_year_other_opex, MONTHS)
    monthly_other_opex_schedule = _compound_with_inflation_and_risk(base_other_opex_schedule)

    monthly_ebitda_schedule = [
        revenue - cogs - staff - opex
        for revenue, cogs, staff, opex in zip(
            monthly_revenue_schedule,
            monthly_cogs_schedule,
            monthly_staff_cost_schedule,
            monthly_other_opex_schedule,
        )
    ]

    accounts_receivable_schedule = [
        revenue * (ACCOUNTS_RECEIVABLE_DAYS / 30.0) for revenue in monthly_revenue_schedule
    ]
    other_assets_schedule = [
        revenue * (OTHER_ASSETS_DAYS / 30.0) for revenue in monthly_revenue_schedule
    ]
    accounts_receivable_other_assets_schedule = [
        ar + other for ar, other in zip(accounts_receivable_schedule, other_assets_schedule)
    ]
    inventory_schedule = [cogs * (INVENTORY_DAYS / 30.0) for cogs in monthly_cogs_schedule]
    accounts_payable_schedule = [
        cost * (ACCOUNTS_PAYABLE_DAYS / 30.0) for cost in monthly_feedstock_cost_schedule
    ]
    other_payables_schedule = [
        (staff + opex) * (OTHER_PAYABLES_DAYS / 30.0)
        for staff, opex in zip(monthly_staff_cost_schedule, monthly_other_opex_schedule)
    ]

    net_working_capital_schedule: List[float] = []
    change_in_nwc_schedule: List[float] = []
    previous_nwc = 0.0
    for ar_other, inventory, ap, other_payable in zip(
        accounts_receivable_other_assets_schedule,
        inventory_schedule,
        accounts_payable_schedule,
        other_payables_schedule,
    ):
        current_nwc = ar_other + inventory - ap - other_payable
        net_working_capital_schedule.append(current_nwc)
        change_in_nwc_schedule.append(current_nwc - previous_nwc)
        previous_nwc = current_nwc

    annual_cassava_tons = _first_year_sum(monthly_cassava_tons)
    annual_ethanol_litres = _first_year_sum(monthly_ethanol_litres)
    annual_animal_feed_ton = _first_year_sum(monthly_animal_feed_ton)
    annual_ethanol_revenue = _first_year_sum(monthly_ethanol_revenue)
    annual_coproduct_revenue = _first_year_sum(monthly_coproduct_revenue)
    annual_revenue = _first_year_sum(monthly_revenue_schedule)
    annual_feedstock_cost_farm = _first_year_sum(farmland_feedstock_cost_schedule)
    annual_feedstock_cost_purchase = _first_year_sum(purchased_feedstock_cost_schedule)
    annual_direct_cost_other = _first_year_sum(monthly_other_direct_cost_schedule)
    annual_direct_cost = _first_year_sum(monthly_cogs_schedule)
    annual_staff_cost = _first_year_sum(monthly_staff_cost_schedule)
    annual_other_opex = _first_year_sum(monthly_other_opex_schedule)
    annual_ebitda = _first_year_sum(monthly_ebitda_schedule)

    depreciation_items: List[Tuple[str, float]] = []
    capex_total = 0.0
    for name, cost, life in CAPEX_ITEMS:
        adjusted_cost = cost
        adjusted_life = life
        if name == "Farm Development":
            adjusted_cost *= farm_share
        capex_total += adjusted_cost
        if adjusted_life:
            depreciation_items.append((name, adjusted_cost / adjusted_life))

    annual_depreciation = sum(amount for _, amount in depreciation_items)
    operating_income = annual_ebitda - annual_depreciation

    interest_expense = LOAN_AMOUNT * LOAN_INTEREST_RATE
    pre_tax_income = operating_income - interest_expense
    taxes = TAX_RATE * pre_tax_income if pre_tax_income > 0 else 0.0
    net_income = pre_tax_income - taxes

    operating_cash_flow = net_income + annual_depreciation
    maintenance_capex = capex_total * MAINTENANCE_CAPEX_RATE
    working_capital_investment = _first_year_sum(change_in_nwc_schedule)
    if net_working_capital_schedule:
        first_year_nwc_index = min(MONTHS_PER_YEAR, len(net_working_capital_schedule)) - 1
        net_working_capital_end = net_working_capital_schedule[first_year_nwc_index]
    else:
        first_year_nwc_index = -1
        net_working_capital_end = 0.0
    free_cash_flow = operating_cash_flow - maintenance_capex - working_capital_investment

    annual_debt_service = 0.0
    if LOAN_INTEREST_RATE > 0:
        factor = (1 + LOAN_INTEREST_RATE) ** LOAN_TERM_YEARS
        annual_debt_service = LOAN_AMOUNT * (LOAN_INTEREST_RATE * factor) / (factor - 1)
    annual_principal_repayment = max(annual_debt_service - interest_expense, 0.0)
    annual_equity_cash_flow = operating_cash_flow - annual_principal_repayment - working_capital_investment

    debt_service_coverage_ratio: Optional[float] = None
    if annual_debt_service > 0:
        debt_service_coverage_ratio = annual_ebitda / annual_debt_service if annual_debt_service else None

    interest_coverage_ratio: Optional[float] = None
    if interest_expense > 0:
        interest_coverage_ratio = operating_income / interest_expense if interest_expense else None

    gross_margin = None
    if annual_revenue:
        gross_profit = annual_revenue - (annual_feedstock_cost_farm + annual_feedstock_cost_purchase + annual_direct_cost_other)
        gross_margin = gross_profit / annual_revenue

    ebitda_margin = (annual_ebitda / annual_revenue) if annual_revenue else None
    net_margin = (net_income / annual_revenue) if annual_revenue else None

    payback_years: Optional[float] = None
    if free_cash_flow > 0:
        payback_years = capex_total / free_cash_flow

    steady_state_index = len(monthly_cassava_tons) - 1
    steady_state = MonthlyRunRate(
        cassava_ton_farm=farmland_tons_schedule[steady_state_index],
        cassava_ton_purchase=purchased_tons_schedule[steady_state_index],
        cassava_ton_total=monthly_cassava_tons[steady_state_index],
        ethanol_litres=monthly_ethanol_litres[steady_state_index],
        animal_feed_ton=monthly_animal_feed_ton[steady_state_index],
        ethanol_revenue=monthly_ethanol_revenue[steady_state_index],
        animal_feed_revenue=monthly_coproduct_revenue[steady_state_index],
        total_revenue=monthly_revenue_schedule[steady_state_index],
        feedstock_cost_farm=farmland_feedstock_cost_schedule[steady_state_index],
        feedstock_cost_purchase=purchased_feedstock_cost_schedule[steady_state_index],
        other_direct_cost=monthly_other_direct_cost_schedule[steady_state_index],
        total_direct_cost=monthly_cogs_schedule[steady_state_index],
        staff_cost=monthly_staff_cost_schedule[steady_state_index],
        other_opex=monthly_other_opex_schedule[steady_state_index],
        ebitda=monthly_ebitda_schedule[steady_state_index],
        ebitda_margin=(
            monthly_ebitda_schedule[steady_state_index] / monthly_revenue_schedule[steady_state_index]
            if monthly_revenue_schedule[steady_state_index]
            else None
        ),
    )

    steady_state_staff_cost = monthly_staff_cost_schedule[steady_state_index]
    steady_state_other_opex = monthly_other_opex_schedule[steady_state_index]

    break_even_ethanol_price: Optional[float] = None
    if steady_state.ethanol_litres > 0:
        numerator = (
            steady_state.total_direct_cost
            + steady_state.staff_cost
            + steady_state.other_opex
            - steady_state.animal_feed_revenue
        )
        break_even_ethanol_price = numerator / steady_state.ethanol_litres

    project_cash_flows = [-capex_total] + [free_cash_flow for _ in range(ANALYSIS_HORIZON_YEARS)]
    project_npv = _net_present_value(project_cash_flows, DISCOUNT_RATE)
    project_irr = _internal_rate_of_return(project_cash_flows)

    monthly_rows: List[List[float | str]] = [
        [
            "Month",
            "Cassava ton (Farm)",
            "Cassava ton (Purchase)",
            "Cassava ton (Total)",
            "Ethanol litres",
            "Animal feed ton",
            "Ethanol revenue",
            "Animal feed revenue",
            "Total revenue",
            "Feedstock cost (Farm)",
            "Feedstock cost (Purchase)",
            "Other direct cost",
            "Total direct cost",
            "Staff costs",
            "Other Opex",
            "EBITDA",
            "Accounts Receivable & Other Assets",
            "Inventory",
            "Accounts Payable",
            "Other Payables",
            "Net Working Capital",
            "Change in Net Working Capital",
        ]
    ]

    for idx, month in enumerate(MONTHS):
        monthly_rows.append(
            [
                month,
                farmland_tons_schedule[idx],
                purchased_tons_schedule[idx],
                monthly_cassava_tons[idx],
                monthly_ethanol_litres[idx],
                monthly_animal_feed_ton[idx],
                monthly_ethanol_revenue[idx],
                monthly_coproduct_revenue[idx],
                monthly_revenue_schedule[idx],
                farmland_feedstock_cost_schedule[idx],
                purchased_feedstock_cost_schedule[idx],
                monthly_other_direct_cost_schedule[idx],
                monthly_cogs_schedule[idx],
                monthly_staff_cost_schedule[idx],
                monthly_other_opex_schedule[idx],
                monthly_ebitda_schedule[idx],
                accounts_receivable_other_assets_schedule[idx],
                inventory_schedule[idx],
                accounts_payable_schedule[idx],
                other_payables_schedule[idx],
                net_working_capital_schedule[idx],
                change_in_nwc_schedule[idx],
            ]
        )

    monthly_rows.append(
        [
            "Annual Total",
            _first_year_sum(farmland_tons_schedule),
            _first_year_sum(purchased_tons_schedule),
            annual_cassava_tons,
            annual_ethanol_litres,
            annual_animal_feed_ton,
            annual_ethanol_revenue,
            annual_coproduct_revenue,
            annual_revenue,
            annual_feedstock_cost_farm,
            annual_feedstock_cost_purchase,
            annual_direct_cost_other,
            annual_direct_cost,
            annual_staff_cost,
            annual_other_opex,
            annual_ebitda,
            accounts_receivable_other_assets_schedule[first_year_nwc_index]
            if first_year_nwc_index >= 0
            else 0.0,
            inventory_schedule[first_year_nwc_index] if first_year_nwc_index >= 0 else 0.0,
            accounts_payable_schedule[first_year_nwc_index] if first_year_nwc_index >= 0 else 0.0,
            other_payables_schedule[first_year_nwc_index] if first_year_nwc_index >= 0 else 0.0,
            net_working_capital_end,
            _first_year_sum(change_in_nwc_schedule),
        ]
    )

    return ScenarioData(
        scenario=scenario_upper,
        farm_share=farm_share,
        steady_state=steady_state,
        monthly_staff_cost=steady_state_staff_cost,
        monthly_other_opex=steady_state_other_opex,
        annual_cassava_tons=annual_cassava_tons,
        annual_ethanol_litres=annual_ethanol_litres,
        annual_animal_feed_ton=annual_animal_feed_ton,
        annual_ethanol_revenue=annual_ethanol_revenue,
        annual_coproduct_revenue=annual_coproduct_revenue,
        annual_revenue=annual_revenue,
        annual_feedstock_cost_farm=annual_feedstock_cost_farm,
        annual_feedstock_cost_purchase=annual_feedstock_cost_purchase,
        annual_direct_cost_other=annual_direct_cost_other,
        annual_direct_cost=annual_direct_cost,
        annual_staff_cost=annual_staff_cost,
        annual_other_opex=annual_other_opex,
        annual_ebitda=annual_ebitda,
        annual_depreciation=annual_depreciation,
        operating_income=operating_income,
        interest_expense=interest_expense,
        pre_tax_income=pre_tax_income,
        taxes=taxes,
        net_income=net_income,
        operating_cash_flow=operating_cash_flow,
        maintenance_capex=maintenance_capex,
        capex_total=capex_total,
        free_cash_flow=free_cash_flow,
        annual_debt_service=annual_debt_service,
        annual_equity_cash_flow=annual_equity_cash_flow,
        debt_service_coverage_ratio=debt_service_coverage_ratio,
        interest_coverage_ratio=interest_coverage_ratio,
        gross_margin=gross_margin,
        ebitda_margin=ebitda_margin,
        net_margin=net_margin,
        break_even_ethanol_price=break_even_ethanol_price,
        payback_years=payback_years,
        project_npv=project_npv,
        project_irr=project_irr,
        working_capital_investment=working_capital_investment,
        net_working_capital_end=net_working_capital_end,
        monthly_cogs_schedule=monthly_cogs_schedule,
        monthly_staff_cost_schedule=monthly_staff_cost_schedule,
        monthly_other_opex_schedule=monthly_other_opex_schedule,
        monthly_revenue_schedule=monthly_revenue_schedule,
        monthly_ebitda_schedule=monthly_ebitda_schedule,
        monthly_accounts_receivable_other_assets_schedule=accounts_receivable_other_assets_schedule,
        monthly_inventory_schedule=inventory_schedule,
        monthly_accounts_payable_schedule=accounts_payable_schedule,
        monthly_other_payables_schedule=other_payables_schedule,
        net_working_capital_schedule=net_working_capital_schedule,
        change_in_net_working_capital_schedule=change_in_nwc_schedule,
        monthly_rows=monthly_rows,
    )


class CassavaFallbackModel:
    """Lightweight workbook generator used when pandas/numpy are unavailable."""

    scenarios: Tuple[str, ...] = tuple(SCENARIOS)

    def _assumptions_rows(self, data: ScenarioData) -> List[List[float | str | None]]:
        capex_breakdown: List[List[float | str | None]] = [["Item", "Cost (USD)", "Life (yrs)"]]
        for name, cost, life in CAPEX_ITEMS:
            adjusted_cost = cost
            if name == "Farm Development":
                adjusted_cost *= data.farm_share
            capex_breakdown.append([name, adjusted_cost, life if life else "N/A"])

        rows: List[List[float | str | None]] = [
            ["Cassava Bioethanol Financial Model"],
            [f"Scenario: {data.scenario}"],
            [None],
            ["Global Assumptions"],
            ["Parameter", "Value", "Units"],
            ["Corporate tax rate", TAX_RATE, "%"],
            ["Discount rate (NPV)", DISCOUNT_RATE, "%"],
            ["Analysis horizon", ANALYSIS_HORIZON_YEARS, "years"],
            ["Loan amount", LOAN_AMOUNT, "USD"],
            ["Loan interest rate", LOAN_INTEREST_RATE, "%"],
            ["Loan term", LOAN_TERM_YEARS, "years"],
            ["Farm share", data.farm_share, "%"],
            ["Steady-state cassava processed", data.steady_state.cassava_ton_total, "ton/month"],
            ["Farm-sourced cassava", data.steady_state.cassava_ton_farm, "ton/month"],
            ["Purchased cassava", data.steady_state.cassava_ton_purchase, "ton/month"],
            ["Ethanol yield", ETHANOL_LITRES_PER_TON, "L/ton"],
            ["Animal feed yield", ANIMAL_FEED_TON_PER_TON, "ton/ton"],
            ["Farm production cost", FARM_PRODUCTION_COST_PER_TON, "USD/ton"],
            ["Purchased cassava cost", PURCHASE_COST_PER_TON, "USD/ton"],
            ["Maintenance capex rate", MAINTENANCE_CAPEX_RATE, "%"],
            ["Other monthly opex", data.monthly_other_opex, "USD"],
            ["Accounts receivable days", ACCOUNTS_RECEIVABLE_DAYS, "days"],
            ["Other asset days", OTHER_ASSETS_DAYS, "days"],
            ["Inventory days", INVENTORY_DAYS, "days"],
            ["Accounts payable days", ACCOUNTS_PAYABLE_DAYS, "days"],
            ["Other payable days", OTHER_PAYABLES_DAYS, "days"],
            [None],
            ["Capital Expenditure"],
        ]

        rows.extend(capex_breakdown)
        rows.extend(
            [
                [None],
                ["Monthly Run Rate (steady-state)"],
                ["Metric", "Value", "Units"],
                ["Ethanol revenue", data.steady_state.ethanol_revenue, "USD"],
                ["Animal feed revenue", data.steady_state.animal_feed_revenue, "USD"],
                ["Total revenue", data.steady_state.total_revenue, "USD"],
                ["Feedstock cost (farm)", data.steady_state.feedstock_cost_farm, "USD"],
                ["Feedstock cost (purchase)", data.steady_state.feedstock_cost_purchase, "USD"],
                ["Other direct cost", data.steady_state.other_direct_cost, "USD"],
                ["Staff costs", data.steady_state.staff_cost, "USD"],
                ["Other opex", data.steady_state.other_opex, "USD"],
                ["EBITDA", data.steady_state.ebitda, "USD"],
                ["EBITDA margin", data.steady_state.ebitda_margin, "%"],
            ]
        )

        rows.extend(
            [
                [None],
                ["Annual Production"],
                ["Cassava processed", data.annual_cassava_tons, "ton"],
                ["Ethanol output", data.annual_ethanol_litres, "L"],
                ["Animal feed output", data.annual_animal_feed_ton, "ton"],
                ["Maintenance capex", data.maintenance_capex, "USD/year"],
            ]
        )

        return rows

    def _income_statement_rows(self, data: ScenarioData) -> List[List[float | str | None]]:
        return [
            ["Annual Summary (USD)"],
            ["Line Item", "Amount"],
            ["Ethanol revenue", data.annual_ethanol_revenue],
            ["Animal feed revenue", data.annual_coproduct_revenue],
            ["Total revenue", data.annual_revenue],
            ["Feedstock costs - farm", data.annual_feedstock_cost_farm],
            ["Feedstock costs - purchase", data.annual_feedstock_cost_purchase],
            ["Other direct costs", data.annual_direct_cost_other],
            ["Total direct costs", data.annual_direct_cost],
            ["Staff costs", data.annual_staff_cost],
            ["Other operating expenses", data.annual_other_opex],
            ["EBITDA", data.annual_ebitda],
            ["Depreciation", data.annual_depreciation],
            ["Operating income (EBIT)", data.operating_income],
            ["Interest expense", data.interest_expense],
            ["Pre-tax income", data.pre_tax_income],
            ["Taxes", data.taxes],
            ["Net income", data.net_income],
            ["Operating cash flow", data.operating_cash_flow],
            ["Change in net working capital", -data.working_capital_investment],
            ["Maintenance capex", -data.maintenance_capex],
            ["Free cash flow", data.free_cash_flow],
            ["Annual debt service", -data.annual_debt_service],
            ["Equity cash flow after debt service", data.annual_equity_cash_flow],
        ]

    def _scenario_summary_rows(self, data: ScenarioData) -> List[List[float | str | None]]:
        def display(value: Optional[float]) -> float | str:
            return value if value is not None else "n/a"

        return [
            ["Key Performance Indicators"],
            ["Metric", "Value", "Units"],
            ["Gross margin", display(data.gross_margin), "%"],
            ["EBITDA margin", display(data.ebitda_margin), "%"],
            ["Net margin", display(data.net_margin), "%"],
            ["Interest coverage", display(data.interest_coverage_ratio), "x"],
            ["Debt service coverage", display(data.debt_service_coverage_ratio), "x"],
            ["Break-even ethanol price", display(data.break_even_ethanol_price), "USD/L"],
            ["Payback period", display(data.payback_years), "years"],
            ["Project NPV", data.project_npv, "USD"],
            ["Project IRR", display(data.project_irr), "%"],
            ["Working capital investment", data.working_capital_investment, "USD"],
            ["Net working capital (year-end)", data.net_working_capital_end, "USD"],
        ]

    def _operating_costs_rows(self, data: ScenarioData) -> List[List[float | str | None]]:
        rows: List[List[float | str | None]] = [["Operating Cost Schedule"], ["Month", "COGS", "Staff Costs", "Other Opex"]]
        for month, cogs, staff, opex in zip(
            MONTHS,
            data.monthly_cogs_schedule,
            data.monthly_staff_cost_schedule,
            data.monthly_other_opex_schedule,
        ):
            rows.append([month, cogs, staff, opex])

        rows.append(
            [
                "Annual Total",
                sum(data.monthly_cogs_schedule),
                sum(data.monthly_staff_cost_schedule),
                sum(data.monthly_other_opex_schedule),
            ]
        )
        return rows

    def _financial_performance_rows(self, data: ScenarioData) -> List[List[float | str | None]]:
        rows: List[List[float | str | None]] = [
            ["Monthly Financial Performance"],
            [
                "Month",
                "Revenue",
                "COGS",
                "Staff Costs",
                "Other Opex",
                "EBITDA",
                "Change in Net Working Capital",
            ],
        ]

        for (
            month,
            revenue,
            cogs,
            staff,
            opex,
            ebitda,
            nwc_change,
        ) in zip(
            MONTHS,
            data.monthly_revenue_schedule,
            data.monthly_cogs_schedule,
            data.monthly_staff_cost_schedule,
            data.monthly_other_opex_schedule,
            data.monthly_ebitda_schedule,
            data.change_in_net_working_capital_schedule,
        ):
            rows.append([month, revenue, cogs, staff, opex, ebitda, nwc_change])

        rows.append(
            [
                "Annual Total",
                sum(data.monthly_revenue_schedule),
                sum(data.monthly_cogs_schedule),
                sum(data.monthly_staff_cost_schedule),
                sum(data.monthly_other_opex_schedule),
                sum(data.monthly_ebitda_schedule),
                sum(data.change_in_net_working_capital_schedule),
            ]
        )

        return rows

    def _working_capital_rows(self, data: ScenarioData) -> List[List[float | str | None]]:
        rows: List[List[float | str | None]] = [
            ["Working Capital Schedule"],
            [
                "Month",
                "Accounts Receivable & Other Assets",
                "Inventory",
                "Accounts Payable",
                "Other Payables",
                "Net Working Capital",
                "Change in Net Working Capital",
            ],
        ]

        for (
            month,
            ar_other,
            inventory,
            accounts_payable,
            other_payables,
            nwc,
            nwc_change,
        ) in zip(
            MONTHS,
            data.monthly_accounts_receivable_other_assets_schedule,
            data.monthly_inventory_schedule,
            data.monthly_accounts_payable_schedule,
            data.monthly_other_payables_schedule,
            data.net_working_capital_schedule,
            data.change_in_net_working_capital_schedule,
        ):
            rows.append(
                [
                    month,
                    ar_other,
                    inventory,
                    accounts_payable,
                    other_payables,
                    nwc,
                    nwc_change,
                ]
            )

        rows.append(
            [
                "Year-end / Change",
                data.monthly_accounts_receivable_other_assets_schedule[-1],
                data.monthly_inventory_schedule[-1],
                data.monthly_accounts_payable_schedule[-1],
                data.monthly_other_payables_schedule[-1],
                data.net_working_capital_schedule[-1],
                sum(data.change_in_net_working_capital_schedule),
            ]
        )

        return rows

    def _monthly_detail_rows(self, data: ScenarioData) -> List[List[float | str | None]]:
        return data.monthly_rows

    def export(self, output: Path | str, *, scenario: str) -> None:
        data = _scenario_parameters(scenario)
        sheets = [
            ("Assumptions", self._assumptions_rows(data)),
            ("Scenario Summary", self._scenario_summary_rows(data)),
            ("Income Statement", self._income_statement_rows(data)),
            ("Financial Performance", self._financial_performance_rows(data)),
            ("Operating Costs", self._operating_costs_rows(data)),
            ("Working Capital", self._working_capital_rows(data)),
            ("Monthly Detail", self._monthly_detail_rows(data)),
        ]
        _write_xlsx(Path(output), sheets)
