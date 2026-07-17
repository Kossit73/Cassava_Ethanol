from types import SimpleNamespace

import pytest

pd = pytest.importorskip("pandas")

import streamlit_app as app
from bioethanol_model.inputs import EditableTable, ProjectionHorizon
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


def test_table_widget_reset_uses_a_new_revision(monkeypatch):
    fake_st = SimpleNamespace(session_state={})
    monkeypatch.setattr(app, "st", fake_st)
    table = EditableTable("Direct Costs Monthly", ["Month", "Amount"], pd.DataFrame())

    original_key = app._table_widget_key(table)
    fake_st.session_state[original_key] = {"edited_rows": {}}

    app._reset_table_widget(table)

    assert original_key not in fake_st.session_state
    assert app._table_editor_revision(table) == 1
    assert app._table_widget_key(table).endswith("_r1")


def test_saved_row_stays_in_draft_until_table_commit(monkeypatch):
    fake_st = SimpleNamespace(session_state={})
    monkeypatch.setattr(app, "st", fake_st)
    table = EditableTable(
        "Direct Costs Monthly",
        ["Month", "Cost Category", "Amount"],
        pd.DataFrame(
            [{"Month": "2025-01", "Cost Category": "Cassava Feedstock", "Amount": 100.0}]
        ),
    )

    draft = app._workspace_draft(table)
    draft.at[0, "Amount"] = 125.0
    stored = app._store_workspace_draft(table, draft, dirty_rows={0})

    assert stored.at[0, "Amount"] == pytest.approx(125.0)
    assert table.data.at[0, "Amount"] == pytest.approx(100.0)
    assert fake_st.session_state[app._workspace_state_key(table, "dirty")] is True

    reset = app._reset_workspace_draft(table)
    assert reset.at[0, "Amount"] == pytest.approx(100.0)


def test_product_routing_row_labels_include_stage_and_output_name():
    table = EditableTable(
        "Product Routing",
        ["Stage Order", "Input Stream", "Output Stream"],
        pd.DataFrame(
            [
                {
                    "Stage Order": 1.0,
                    "Input Stream": "Cassava",
                    "Output Stream": "Fuel Ethanol",
                },
                {
                    "Stage Order": 2,
                    "Input Stream": "Starch Pool",
                    "Output Stream": "Dextrin",
                },
            ]
        ),
    )

    assert app._format_workspace_row(table, table.data, 0) == "1.1. Fuel Ethanol"
    assert app._format_workspace_row(table, table.data, 1) == "2.2. Dextrin"


def test_workspace_validation_rejects_duplicate_month_and_negative_production():
    table = EditableTable(
        "Production Monthly",
        ["Start Month", "Cassava ton", "Ethanol litres", "Animal Feed ton", "Growth %"],
        pd.DataFrame(
            [
                {
                    "Start Month": "2025-01",
                    "Cassava ton": 100.0,
                    "Ethanol litres": 0.0,
                    "Animal Feed ton": 0.0,
                    "Growth %": 0.0,
                },
                {
                    "Start Month": "2025-01",
                    "Cassava ton": -5.0,
                    "Ethanol litres": 0.0,
                    "Animal Feed ton": 0.0,
                    "Growth %": 0.0,
                },
            ]
        ),
    )

    errors = app._validate_workspace_draft(
        table,
        table.data,
        ProjectionHorizon(start_year=2025, end_year=2027),
    )

    assert any("Duplicate effective-period" in error for error in errors)
    assert any("cannot be negative" in error for error in errors)
