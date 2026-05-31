import pytest

pd = pytest.importorskip("pandas")

from streamlit_app import _set_dataframe_cell
from streamlit_app import (
    _annual_decimal_to_monthly_percent,
    _build_production_horizon_seed,
    _monthly_percent_input_to_annual_decimal,
)


def test_set_dataframe_cell_upcasts_for_incompatible_editor_value():
    df = pd.DataFrame({"Start Month": [202501]})

    _set_dataframe_cell(df, 0, "Start Month", None)

    assert df.at[0, "Start Month"] is None
    assert df["Start Month"].dtype == "object"


def test_monthly_percent_conversion_round_trips_to_annual_decimal():
    annual_decimal = _monthly_percent_input_to_annual_decimal(2.0)

    assert annual_decimal == pytest.approx(0.24)
    assert _annual_decimal_to_monthly_percent(annual_decimal) == pytest.approx(2.0)


def test_build_production_horizon_seed_sets_only_anchor_month():
    seed_df, anchor_period = _build_production_horizon_seed(
        ["Start Month", "Cassava ton", "Ethanol litres", "Animal Feed ton", "Growth %"],
        start_year=2025,
        end_year=2025,
        planning_start="2025-03",
        first_month_cassava_ton=12000.0,
        monthly_increment_percent=1.5,
    )

    assert anchor_period.strftime("%Y-%m") == "2025-03"
    assert len(seed_df) == 12

    anchor_row = seed_df.loc[seed_df["Start Month"] == "2025-03"].iloc[0]
    assert float(anchor_row["Cassava ton"]) == pytest.approx(12000.0)
    assert float(anchor_row["Growth %"]) == pytest.approx(0.18)

    non_anchor = seed_df.loc[seed_df["Start Month"] != "2025-03"]
    assert non_anchor["Cassava ton"].isna().all()
