import pytest

pd = pytest.importorskip("pandas")
pdt = pytest.importorskip("pandas.testing")

from bioethanol_model.financial_model import CassavaBioethanolModel
from bioethanol_model.schedules import compute_production_tables


def test_direct_costs_preserved_across_scenarios():
    model = CassavaBioethanolModel()

    direct_rows = pd.DataFrame(
        [
            {"Month": "2025-01", "Cost Category": "Cassava Feedstock", "Amount": 575_000.0},
            {"Month": "2025-06", "Cost Category": "Enzymes & Chemicals", "Amount": 210_000.0},
            {"Month": "2025-09", "Cost Category": "Energy Cost", "Amount": 195_000.0},
        ]
    )

    table = model.input_page.direct_costs_monthly
    table.set_data(direct_rows, mark_user_input=True)

    expected = direct_rows.reset_index(drop=True)

    for scenario in model.SCENARIOS:
        prepared = model._prepare_page_for_scenario(scenario)
        result = prepared.direct_costs_monthly.data.reset_index(drop=True)

        pdt.assert_frame_equal(result, expected, check_dtype=False)
        assert not prepared.direct_costs_monthly.placeholder


def test_production_annual_rollup_avoids_pandas_year_alias():
    production = pd.DataFrame(
        [
            {"Start Month": "2025-01", "Cassava ton": 10_000.0, "Growth %": 0.0},
            {"Start Month": "2026-01", "Cassava ton": 12_000.0, "Growth %": 0.0},
        ]
    )

    result = compute_production_tables(production, 2025, 2026)

    assert list(result.annual.index) == [2025, 2026]
    assert result.annual.loc[2025, "Cassava ton"] == pytest.approx(120_000.0)
    assert result.annual.loc[2026, "Cassava ton"] == pytest.approx(144_000.0)


def test_build_scales_debt_to_reduced_capex_envelope():
    model = CassavaBioethanolModel()
    page = model.input_page
    for table in page.tables().values():
        table.set_data(table.data, mark_user_input=True)

    initial_investment = page.initial_investment.data.copy()
    initial_investment["Cost"] = [1_000_000.0, 0.0, 0.0, 0.0, 0.0]
    page.initial_investment.set_data(initial_investment, mark_user_input=True)

    result = model.build("FARM_ONLY")
    metrics = result["metrics"]
    loan_summary = result["loan_schedule"].summary
    adjusted_draw = float(pd.to_numeric(loan_summary["Draw"], errors="coerce").fillna(0.0).sum())

    assert adjusted_draw == pytest.approx(1_000_000.0)
    assert metrics["Debt Funding Original Draw"] == pytest.approx(24_000_000.0)
    assert metrics["Debt Funding Adjusted Draw"] == pytest.approx(1_000_000.0)
    assert metrics["Debt Funding Reduction"] == pytest.approx(23_000_000.0)
    assert metrics["Initial Loan Funding"] <= metrics["Total Initial Investment"] + 1e-6


def test_cassava_feedstock_cost_is_derived_from_tonnage_and_price():
    model = CassavaBioethanolModel()
    page = model.input_page
    for table in page.tables().values():
        table.set_data(table.data, mark_user_input=True)

    globals_df = page.global_inputs.data.copy()
    globals_df.loc[globals_df["Parameter"] == "Contracted feedstock share", "Value"] = 0.0
    globals_df.loc[globals_df["Parameter"] == "Contract feedstock discount", "Value"] = 0.0
    globals_df.loc[globals_df["Parameter"] == "Cassava farm cost per ton", "Value"] = 45.0
    globals_df.loc[globals_df["Parameter"] == "Cassava purchase cost per ton", "Value"] = 70.0
    globals_df.loc[globals_df["Parameter"] == "Hybrid farm share", "Value"] = 0.5
    page.global_inputs.set_data(globals_df, mark_user_input=True)

    expected_by_scenario = {
        "FARM_ONLY": 10_000.0 * 45.0,
        "BUY_ONLY": 10_000.0 * 70.0,
        "HYBRID": 10_000.0 * (0.5 * 45.0 + 0.5 * 70.0),
    }
    for scenario, expected in expected_by_scenario.items():
        model.clear_cache()
        result = model.build(scenario)
        direct_monthly = result["costs"]["Direct Costs"].monthly
        january_value = float(direct_monthly.loc[pd.Timestamp("2025-01-01"), "Cassava Feedstock"])
        multiplier = float(result["metrics"].get("Commercial Cost Multiplier", 1.0))
        assert january_value == pytest.approx(expected * multiplier)
