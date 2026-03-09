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
    assert not schedule.empty
    assert (pd.to_numeric(schedule["Interest Rate"], errors="coerce") <= 0.08).all()


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
