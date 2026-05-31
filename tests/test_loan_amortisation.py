import pytest

pd = pytest.importorskip("pandas")

from bioethanol_model.schedules import compute_loan_schedule


def _monthly_interest_column(annual_df):
    return next(col for col in annual_df.columns if str(col).startswith("Monthly Interest"))


def test_yearly_amortisation_aggregates_monthly_schedule():
    loan_inputs = pd.DataFrame(
        [
            {
                "Loan": "Facility A",
                "Loan Amount": 120_000.0,
                "Tenor Years": 2,
                "Grace Years": 0,
                "Interest Rate": 0.12,
                "Amortization": "Straight",
                "Start Month": "2025-01",
            }
        ]
    )

    output = compute_loan_schedule(loan_inputs, 2025, 2026)
    schedule = output.schedule
    annual = output.annual

    assert not schedule.empty
    assert not annual.empty

    year_2025 = annual.loc[annual["Year"] == 2025].iloc[0]
    monthly_2025 = schedule.loc[schedule["Month"].dt.year == 2025]
    monthly_interest_col = _monthly_interest_column(annual)

    assert pytest.approx(year_2025["Interest Rate"], rel=1e-9) == 0.12
    assert pytest.approx(year_2025["Yearly Remaining Balance"], rel=1e-9) == 120_000.0
    assert pytest.approx(year_2025[monthly_interest_col], rel=1e-9) == (
        year_2025["Yearly Remaining Balance"] * year_2025["Interest Rate"] / 12.0
    )

    assert pytest.approx(monthly_2025["Interest"].sum(), rel=1e-9) == year_2025["Interest Paid"]
    assert pytest.approx(monthly_2025["Principal"].sum(), rel=1e-9) == year_2025["Principal Paid"]


def test_percentage_interest_rates_are_normalised():
    loan_inputs = pd.DataFrame(
        [
            {
                "Loan": "Senior Debt",
                "Loan Amount": 1_250_500.0,
                "Tenor Years": 3,
                "Grace Years": 2,
                "Interest Rate": 7.5,
                "Amortization": "Straight",
                "Start Month": "2025-01",
            }
        ]
    )

    output = compute_loan_schedule(loan_inputs, 2025, 2027)
    annual = output.annual

    first_year = annual.loc[annual["Year"] == 2025].iloc[0]
    monthly_interest_col = _monthly_interest_column(annual)
    assert pytest.approx(first_year["Interest Rate"], rel=1e-9) == 0.075
    expected_monthly_interest = first_year["Yearly Remaining Balance"] * 0.075 / 12.0
    assert pytest.approx(first_year[monthly_interest_col], rel=1e-9) == expected_monthly_interest
    assert pytest.approx(first_year["Interest Paid"], rel=1e-6) == expected_monthly_interest * 12


def test_pre_projection_loan_start_carries_opening_balance_without_redraw():
    loan_inputs = pd.DataFrame(
        [
            {
                "Loan": "Legacy Debt",
                "Loan Amount": 120_000.0,
                "Tenor Years": 3,
                "Grace Years": 0,
                "Interest Rate": 0.0,
                "Amortization": "Straight",
                "Start Month": "2024-01",
            }
        ]
    )

    output = compute_loan_schedule(loan_inputs, 2025, 2027)
    jan_2025 = output.schedule.loc[output.schedule["Month"] == pd.Timestamp("2025-01-01")].iloc[0]

    assert float(jan_2025["Draw"]) == pytest.approx(0.0)
    assert float(jan_2025["Opening Balance"]) == pytest.approx(80_000.0, rel=1e-6)
