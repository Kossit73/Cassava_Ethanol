import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from bioethanol_model.increment import apply_yearly_increment
from bioethanol_model.inputs import default_input_page
from bioethanol_model.schedules import compute_cost_tables


def test_yearly_increment_updates_cost_tables():
    page = default_input_page()

    direct_df = page.direct_costs_monthly.data.copy()
    cassava_idx = direct_df.index[direct_df["Cost Category"] == "Cassava Feedstock"][0]

    updated_direct = apply_yearly_increment(
        direct_df,
        cassava_idx,
        date_column="Month",
        frequency="M",
        value_columns=["Amount"],
        increments={"Amount": 0.1},
        match_columns=["Cost Category"],
        horizon_end=f"{page.projection.end_year:04d}-12",
    )
    page.direct_costs_monthly.set_data(updated_direct, mark_user_input=True)
    inflation_df = page.inflation_schedule.model_frame.copy()
    inflation_df["CPI"] = 0.0

    cost_outputs = compute_cost_tables(
        page.direct_costs_monthly.model_frame,
        page.staff_costs_monthly.model_frame,
        page.other_opex_monthly.model_frame,
        inflation_df,
        page.projection.start_year,
        page.projection.end_year,
    )

    monthly_direct = cost_outputs["Direct Costs"].monthly

    january_2025 = monthly_direct.loc[pd.Timestamp("2025-01-01"), "Cassava Feedstock"]
    january_2026 = monthly_direct.loc[pd.Timestamp("2026-01-01"), "Cassava Feedstock"]
    january_2027 = monthly_direct.loc[pd.Timestamp("2027-01-01"), "Cassava Feedstock"]

    assert january_2025 == pytest.approx(600_000.0)
    assert january_2026 == pytest.approx(600_000.0 * 1.1)
    assert january_2027 == pytest.approx(600_000.0 * (1.1**2))
