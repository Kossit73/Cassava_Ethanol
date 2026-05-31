import pytest

pd = pytest.importorskip("pandas")

from bioethanol_model.schedules import (
    compute_cost_tables,
    compute_production_tables,
    compute_revenue_schedule,
)


def test_revenue_cpi_compounds_from_planning_start():
    production_inputs = pd.DataFrame(
        [{"Start Month": "2025-01", "Cassava ton": 1_000.0, "Growth %": 0.0}]
    )
    production = compute_production_tables(production_inputs, 2025, 2027, planning_start="2025-01")

    revenue_inputs = pd.DataFrame(
        [{"Product": "Ethanol", "Base Price": 1.0, "Escalation": 0.0}]
    )
    inflation = pd.DataFrame(
        [
            {"Year": 2025, "CPI": 0.12},
            {"Year": 2026, "CPI": 0.12},
        ]
    )

    revenue = compute_revenue_schedule(
        production,
        revenue_inputs,
        inflation,
        planning_start="2025-01",
    )
    implied_price = revenue.monthly["Ethanol revenue"] / production.monthly["Ethanol litres"]

    assert float(implied_price.loc[pd.Timestamp("2025-01-01")]) == pytest.approx(1.0, rel=1e-6)
    assert float(implied_price.loc[pd.Timestamp("2026-01-01")]) == pytest.approx(1.12, rel=1e-4)
    assert float(implied_price.loc[pd.Timestamp("2027-01-01")]) == pytest.approx(1.12 * 1.12, rel=1e-4)


def test_cost_cpi_compounds_from_planning_start():
    direct = pd.DataFrame(
        [{"Month": "2025-01", "Cost Category": "Cassava Feedstock", "Amount": 100.0}]
    )
    staff = pd.DataFrame(columns=["Month", "Department", "Cost"])
    other = pd.DataFrame(columns=["Month", "Category", "Amount"])
    inflation = pd.DataFrame(
        [
            {"Year": 2025, "CPI": 0.12},
            {"Year": 2026, "CPI": 0.12},
        ]
    )

    outputs = compute_cost_tables(
        direct,
        staff,
        other,
        inflation,
        2025,
        2027,
        planning_start="2025-01",
    )
    monthly_direct = outputs["Direct Costs"].monthly["Cassava Feedstock"]

    assert float(monthly_direct.loc[pd.Timestamp("2025-01-01")]) == pytest.approx(100.0, rel=1e-6)
    assert float(monthly_direct.loc[pd.Timestamp("2026-01-01")]) == pytest.approx(112.0, rel=1e-4)
    assert float(monthly_direct.loc[pd.Timestamp("2027-01-01")]) == pytest.approx(125.44, rel=1e-4)
