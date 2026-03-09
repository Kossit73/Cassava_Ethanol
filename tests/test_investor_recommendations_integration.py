import copy

import pytest

pd = pytest.importorskip("pandas")

from bioethanol_model import CassavaBioethanolModel
from bioethanol_model.inputs import default_input_page


def _set_global(page, parameter: str, value: float) -> None:
    df = page.global_inputs.data.copy()
    mask = df["Parameter"] == parameter
    if not mask.any():
        df = pd.concat(
            [df, pd.DataFrame([{"Parameter": parameter, "Value": value, "Units": ""}])],
            ignore_index=True,
        )
    else:
        df.loc[mask, "Value"] = value
    page.global_inputs.set_data(df, mark_user_input=True)


def test_build_blocks_when_required_inputs_missing() -> None:
    page = default_input_page()
    page.global_inputs.set_data(page.global_inputs.data, mark_user_input=True)
    page.revenue_inputs.set_data(pd.DataFrame(columns=page.revenue_inputs.columns), mark_user_input=True)
    model = CassavaBioethanolModel(copy.deepcopy(page))

    with pytest.raises(ValueError, match="Missing required input tables"):
        model.build("FARM_ONLY")


def test_risk_and_commercial_metrics_are_exposed() -> None:
    page = default_input_page()
    # enable all required tables
    for table in page.tables().values():
        table.set_data(table.data, mark_user_input=True)

    _set_global(page, "Offtake floor price (USD/L)", 0.6)
    _set_global(page, "Offtake ceiling price (USD/L)", 1.0)
    _set_global(page, "Take-or-pay share", 0.9)
    _set_global(page, "Contracted feedstock share", 0.7)
    _set_global(page, "Contract feedstock discount", 0.1)

    model = CassavaBioethanolModel(copy.deepcopy(page))
    result = model.build("FARM_ONLY")
    metrics = result["metrics"]

    assert "Risk Score" in metrics
    assert "Commercial Cost Multiplier" in metrics
    assert "Risk Volume Stress (EV)" in metrics
    assert "Risk Schedule Stress (P90)" in metrics
    assert "Minimum Monthly Cash Balance" in metrics
    assert "Peak Funding Requirement" in metrics
    assert "Interest Coverage (avg annual)" in metrics
    assert "Net Debt / EBITDA (avg annual)" in metrics
    assert "Working Capital Days (CCC avg)" in metrics
    assert 0 <= float(metrics["Risk Score"]) <= 1


def test_debt_strategy_toggles_adjust_schedule() -> None:
    page = default_input_page()
    for table in page.tables().values():
        table.set_data(table.data, mark_user_input=True)

    _set_global(page, "Debt sculpting enabled", 1.0)
    _set_global(page, "Refinancing enabled", 1.0)
    _set_global(page, "Refinancing interest rate", 0.05)

    model = CassavaBioethanolModel(copy.deepcopy(page))
    result = model.build("FARM_ONLY")

    schedule = result["loan_schedule"].schedule
    metrics = result["metrics"]
    assert not schedule.empty
    assert (pd.to_numeric(schedule["Interest Rate"], errors="coerce") <= 0.08).all()
    assert "Cash Sweep Applied" in metrics
    assert "DSRA Reset Amount" in metrics


def test_validation_rejects_offtake_corridor_inversion() -> None:
    page = default_input_page()
    for table in page.tables().values():
        table.set_data(table.data, mark_user_input=True)

    _set_global(page, "Offtake floor price (USD/L)", 1.0)
    _set_global(page, "Offtake ceiling price (USD/L)", 0.8)

    model = CassavaBioethanolModel(copy.deepcopy(page))
    with pytest.raises(ValueError, match="ceiling price"):
        model.build("FARM_ONLY")


def test_validation_rejects_loan_start_outside_projection() -> None:
    page = default_input_page()
    for table in page.tables().values():
        table.set_data(table.data, mark_user_input=True)

    loan = page.loan_schedule.data.copy()
    loan.loc[:, "Start Month"] = "2040-01"
    page.loan_schedule.set_data(loan, mark_user_input=True)

    model = CassavaBioethanolModel(copy.deepcopy(page))
    with pytest.raises(ValueError, match="projection window"):
        model.build("FARM_ONLY")


def test_validation_allows_loan_start_before_projection_start() -> None:
    page = default_input_page()
    page.projection.start_year = 2025
    page.projection.end_year = 2034
    for table in page.tables().values():
        table.set_data(table.data, mark_user_input=True)

    loan = page.loan_schedule.data.copy()
    loan.loc[:, "Start Month"] = "2024-01"
    page.loan_schedule.set_data(loan, mark_user_input=True)

    model = CassavaBioethanolModel(copy.deepcopy(page))
    result = model.build("FARM_ONLY")
    assert "metrics" in result


def test_validation_allows_tornado_like_share_variation() -> None:
    page = default_input_page()
    for table in page.tables().values():
        table.set_data(table.data, mark_user_input=True)

    _set_global(page, "Investor share capital", 0.495)
    _set_global(page, "Owner share capital", 0.55)

    model = CassavaBioethanolModel(copy.deepcopy(page))
    result = model.build("FARM_ONLY")
    metrics = result["metrics"]
    assert "Investor Share" in metrics and "Owner Share" in metrics
    assert abs(float(metrics["Investor Share"]) + float(metrics["Owner Share"]) - 1.0) < 1e-6


def test_percent_unit_normalization_accepts_whole_percent_inputs() -> None:
    page = default_input_page()
    for table in page.tables().values():
        table.set_data(table.data, mark_user_input=True)

    _set_global(page, "Discount rate", 12.0)
    _set_global(page, "Corporate tax rate", 28.0)

    model = CassavaBioethanolModel(copy.deepcopy(page))
    result = model.build("FARM_ONLY")
    metrics = result["metrics"]
    audit = result.get("assumption_quality_audit", {})

    assert abs(float(metrics["Discount Rate"]) - 0.12) < 1e-9
    assert abs(float(metrics["Corporate Tax Rate"]) - 0.28) < 1e-9
    assert bool(audit.get("passed"))
    assert "Discount rate" in set(audit.get("converted", []))


def test_accounting_invariants_metrics_are_exposed_and_consistent() -> None:
    page = default_input_page()
    for table in page.tables().values():
        table.set_data(table.data, mark_user_input=True)

    model = CassavaBioethanolModel(copy.deepcopy(page))
    result = model.build("FARM_ONLY")
    metrics = result["metrics"]

    assert "Invariant Balance Sheet Balanced" in metrics
    assert "Invariant Cash Flow Bridge Consistent" in metrics
    assert "Invariant Debt Rollforward Consistent" in metrics
    assert float(metrics["Invariant Balance Sheet Balanced"]) == 1.0
    assert float(metrics["Invariant Cash Flow Bridge Consistent"]) == 1.0
    assert float(metrics["Invariant Debt Rollforward Consistent"]) == 1.0


def test_financial_metrics_regression_snapshot_smoke() -> None:
    page = default_input_page()
    for table in page.tables().values():
        table.set_data(table.data, mark_user_input=True)

    model_a = CassavaBioethanolModel(copy.deepcopy(page))
    metrics_a = model_a.build("FARM_ONLY")["metrics"]
    model_b = CassavaBioethanolModel(copy.deepcopy(page))
    metrics_b = model_b.build("FARM_ONLY")["metrics"]

    snapshot_a = {
        "Project NPV": round(float(metrics_a["Project NPV"]), 2),
        "Project IRR": round(float(metrics_a["Project IRR"]), 6),
        "DSCR (min)": round(float(metrics_a.get("DSCR (min)", 0.0)), 6),
        "Simple Payback (years)": round(float(metrics_a["Simple Payback (years)"]), 6),
    }
    snapshot_b = {
        "Project NPV": round(float(metrics_b["Project NPV"]), 2),
        "Project IRR": round(float(metrics_b["Project IRR"]), 6),
        "DSCR (min)": round(float(metrics_b.get("DSCR (min)", 0.0)), 6),
        "Simple Payback (years)": round(float(metrics_b["Simple Payback (years)"]), 6),
    }
    assert snapshot_a == snapshot_b


def test_build_handles_zero_volume_months_edge_case() -> None:
    page = default_input_page()
    for table in page.tables().values():
        table.set_data(table.data, mark_user_input=True)

    prod = page.production_monthly.data.copy()
    if "Ethanol litres" in prod.columns:
        prod["Ethanol litres"] = 0.0
    page.production_monthly.set_data(prod, mark_user_input=True)

    model = CassavaBioethanolModel(copy.deepcopy(page))
    result = model.build("FARM_ONLY")
    metrics = result["metrics"]

    assert "Project NPV" in metrics
    assert float(metrics["Invariant Balance Sheet Balanced"]) == 1.0


def test_refinancing_year_outside_horizon_is_ignored() -> None:
    page = default_input_page()
    for table in page.tables().values():
        table.set_data(table.data, mark_user_input=True)

    _set_global(page, "Refinancing enabled", 1.0)
    _set_global(page, "Refinancing year", 2100.0)

    model = CassavaBioethanolModel(copy.deepcopy(page))
    result = model.build("FARM_ONLY")
    metrics = result["metrics"]

    assert "Refinancing Costs" in metrics
    assert float(metrics["Refinancing Costs"]) == 0.0


def test_build_handles_empty_risk_schedule() -> None:
    page = default_input_page()
    for table in page.tables().values():
        table.set_data(table.data, mark_user_input=True)

    empty_risk = pd.DataFrame(columns=page.risk_schedule.columns)
    page.risk_schedule.set_data(empty_risk, mark_user_input=True)

    model = CassavaBioethanolModel(copy.deepcopy(page))
    result = model.build("FARM_ONLY")
    metrics = result["metrics"]

    assert "Risk Score" in metrics
    assert float(metrics["Risk Score"]) == 0.0
