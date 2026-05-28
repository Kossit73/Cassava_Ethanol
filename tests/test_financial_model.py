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


def test_global_assumptions_are_auto_balanced_and_synced():
    model = CassavaBioethanolModel()
    page = model.input_page
    for table in page.tables().values():
        table.set_data(table.data, mark_user_input=True)

    globals_df = page.global_inputs.data.copy()
    globals_df.loc[globals_df["Parameter"] == "Investor share capital", "Value"] = 0.495
    globals_df.loc[globals_df["Parameter"] == "Owner share capital", "Value"] = 0.55
    globals_df.loc[globals_df["Parameter"] == "Contracted feedstock share", "Value"] = 0.7
    globals_df.loc[globals_df["Parameter"] == "Open market feedstock share", "Value"] = 0.9
    page.global_inputs.set_data(globals_df, mark_user_input=True)

    result = model.build("FARM_ONLY")
    snapshot_globals = result["input_page_snapshot"].global_inputs.model_frame
    lookup = snapshot_globals.set_index("Parameter")["Value"].to_dict()

    investor = float(lookup["Investor share capital"])
    owner = float(lookup["Owner share capital"])
    contracted = float(lookup["Contracted feedstock share"])
    open_market = float(lookup["Open market feedstock share"])

    assert investor + owner == pytest.approx(1.0)
    assert contracted == pytest.approx(0.7)
    assert open_market == pytest.approx(0.3)
    assert result["metrics"]["Automation Adjustments Applied"] >= 1.0


def test_production_annual_is_auto_synced_from_monthly():
    model = CassavaBioethanolModel()
    page = model.input_page
    for table in page.tables().values():
        table.set_data(table.data, mark_user_input=True)

    stale_annual = page.production_annual.data.copy()
    stale_annual.loc[:, "Year"] = 2035
    page.production_annual.set_data(stale_annual, mark_user_input=True)

    result = model.build("FARM_ONLY")
    synced = result["input_page_snapshot"].production_annual.model_frame

    assert not synced.empty
    assert synced["Year"].min() >= page.projection.start_year
    assert synced["Year"].max() <= page.projection.end_year
    jan_row = synced.loc[synced["Year"] == 2025].iloc[0]
    assert float(jan_row["Cassava ton"]) == pytest.approx(120_000.0)
    assert result["metrics"]["Automation Production Annual Rows"] >= 1.0


def test_corporate_tax_rate_propagates_to_tax_schedule_and_metrics():
    model = CassavaBioethanolModel()
    page = model.input_page
    for table in page.tables().values():
        table.set_data(table.data, mark_user_input=True)

    globals_df = page.global_inputs.data.copy()
    globals_df.loc[globals_df["Parameter"] == "Corporate tax rate", "Value"] = 0.33
    page.global_inputs.set_data(globals_df, mark_user_input=True)

    result = model.build("FARM_ONLY")
    snapshot = result["input_page_snapshot"]
    tax_df = snapshot.tax_schedule.model_frame
    mask = tax_df["Item"].astype(str).str.contains("corporate income tax", case=False, na=False)
    synced_tax = float(pd.to_numeric(tax_df.loc[mask, "Base Rate"], errors="coerce").iloc[0])

    assert float(result["metrics"]["Corporate Tax Rate"]) == pytest.approx(0.33)
    assert synced_tax == pytest.approx(0.33)
    assert result["metrics"]["Automation Tax Schedule Rows Synced"] >= 1.0
