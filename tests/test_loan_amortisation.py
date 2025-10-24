import pytest

pd = pytest.importorskip("pandas")

from bioethanol_model.schedules import compute_loan_schedule


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

    assert pytest.approx(year_2025["Interest Rate"], rel=1e-9) == 0.12
    assert pytest.approx(year_2025["Yearly Remaining Balance"], rel=1e-9) == 120_000.0
    assert pytest.approx(year_2025["Monthly Interest (Balance × Rate / 12)"], rel=1e-9) == (
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
    assert pytest.approx(first_year["Interest Rate"], rel=1e-9) == 0.075
    expected_monthly_interest = first_year["Yearly Remaining Balance"] * 0.075 / 12.0
    assert pytest.approx(first_year["Monthly Interest (Balance × Rate / 12)"], rel=1e-9) == expected_monthly_interest
    assert pytest.approx(first_year["Interest Paid"], rel=1e-6) == expected_monthly_interest * 12
