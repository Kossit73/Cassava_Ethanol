import pytest

pd = pytest.importorskip("pandas")

from bioethanol_model.schedules import (
    CostOutput,
    RevenueOutput,
    compute_working_capital,
)


def _monthly_index(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=periods, freq="M")


def test_working_capital_annual_aligned_to_projection_horizon():
    months = _monthly_index("2026-01-31", 24)

    revenue_monthly = pd.DataFrame({"Total Revenue": 1000.0}, index=months)
    revenue_annual = revenue_monthly.groupby(revenue_monthly.index.year).sum()
    revenue_annual.index.name = "Year"
    revenue = RevenueOutput(revenue_monthly, revenue_annual)

    direct_monthly = pd.DataFrame({"Cassava": 400.0}, index=months)
    staff_monthly = pd.DataFrame({"Dept": 200.0}, index=months)
    other_monthly = pd.DataFrame({"Other": 100.0}, index=months)

    def _annual(df: pd.DataFrame) -> pd.DataFrame:
        annual = df.groupby(df.index.year).sum()
        annual.index.name = "Year"
        return annual

    cost_outputs = {
        "Direct Costs": CostOutput(direct_monthly, _annual(direct_monthly)),
        "Staff Costs": CostOutput(staff_monthly, _annual(staff_monthly)),
        "Other Opex": CostOutput(other_monthly, _annual(other_monthly)),
    }

    metrics_input = pd.DataFrame(
        {
            "Metric": [
                "Receivables days",
                "Inventory days",
                "Prepaid expense days",
                "Other assets percent of revenue",
                "Payables days",
                "Other payable days",
            ],
            "Value": [30, 45, 10, 0.10, 20, 15],
            "Effective Month": ["2026-01"] * 6,
        }
    )

    working_capital = compute_working_capital(
        revenue,
        cost_outputs,
        accounts_receivable_inputs=metrics_input,
        inventory_inputs=pd.DataFrame(),
    )

    annual = working_capital.annual
    assert list(annual.index) == [2026, 2027]

    monthly = working_capital.monthly
    expected_year_end = monthly.loc["2026-12-31"]
    pd.testing.assert_series_equal(
        annual.loc[2026],
        expected_year_end,
        check_names=False,
        check_dtype=False,
    )

    expected_net_wc = (
        1000.0
        + 700.0 * (45 / 30.0)
        + 300.0 * (10 / 30.0)
        + 1000.0 * 0.10
        - 700.0 * (20 / 30.0)
        - 300.0 * (15 / 30.0)
    )

    assert pytest.approx(expected_net_wc) == annual.loc[2026, "Net Working Capital"]
