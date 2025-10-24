import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from bioethanol_model.financial_model import CassavaBioethanolModel
from bioethanol_model.schedules import _derive_initial_investment_components


def test_initial_investment_metrics_align_with_cashflows():
    model = CassavaBioethanolModel.default()
    results = model.build()
    metrics = results["metrics"]
    financials = results["financials"]
    revenue = results["revenue"]

    (
        derived_outlay,
        derived_loan,
        adjusted_free_cf,
        _,
    ) = _derive_initial_investment_components(financials, revenue)

    assert metrics["Initial Project Outlay"] == pytest.approx(derived_outlay)
    assert metrics["Initial Loan Funding"] == pytest.approx(derived_loan)
    assert metrics["Initial Equity Investment"] == pytest.approx(
        derived_outlay - derived_loan
    )

    payback = results["payback"]
    assert not payback.empty
    expected_first_month = adjusted_free_cf.iloc[0] - derived_outlay
    assert payback["Cumulative FCF"].iloc[0] == pytest.approx(expected_first_month)
