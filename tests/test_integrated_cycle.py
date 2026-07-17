from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bioethanol_model.exporter import _write_integrated_cycle_pages
from bioethanol_model.financial_model import CassavaBioethanolModel
from bioethanol_model.inputs import InputLandingPage, default_input_page


@pytest.fixture(scope="module")
def integrated_results():
    model = CassavaBioethanolModel.default()
    results = {scenario: model.build(scenario) for scenario in model.SCENARIOS}
    return model, results


def test_cycle_anchor_expands_annually_and_processing_is_three_months(integrated_results):
    _, results = integrated_results
    output = results["HYBRID"]["integrated_cycle"]

    assert output is not None
    cycles = output.cycle_summary
    assert list(cycles["Year"].head(3)) == [2025, 2026, 2027]
    assert set(cycles["Cultivation Months"]) == {9}
    assert set(cycles["Processing Months"]) == {3}
    assert cycles.loc[cycles["Year"] == 2026, "Cassava Processing Target ton"].iloc[0] == pytest.approx(
        110_000.0 * 1.04
    )

    for _, cycle in cycles.iterrows():
        planting = pd.Period(cycle["Planting Month"], freq="M")
        harvest = pd.Period(cycle["Harvest Month"], freq="M")
        processing_periods = [
            pd.Period(value, freq="M") for value in cycle["Processing Month List"]
        ]
        assert harvest == planting + 8
        assert len(processing_periods) == 3
        assert processing_periods[0] == harvest + 1
        active = output.monthly_physical.loc[
            [period.to_timestamp() for period in processing_periods],
            "Cassava Processed ton",
        ]
        assert (active > 0).all()


def test_farm_buy_and_hybrid_sourcing_are_physically_distinct(integrated_results):
    _, results = integrated_results
    farm = results["FARM_ONLY"]["integrated_cycle"]
    buy = results["BUY_ONLY"]["integrated_cycle"]
    hybrid = results["HYBRID"]["integrated_cycle"]

    assert farm.monthly_physical["Farm Cassava Delivered ton"].sum() > 0
    assert farm.monthly_physical["Purchased Cassava Delivered ton"].sum() == pytest.approx(0.0)
    assert buy.monthly_physical["Farm Cassava Delivered ton"].sum() == pytest.approx(0.0)
    assert buy.monthly_physical["Purchased Cassava Delivered ton"].sum() > 0
    assert buy.farm_monthly["Total Farm Operating Cost"].sum() == pytest.approx(0.0)
    assert hybrid.monthly_physical["Farm Cassava Delivered ton"].sum() > 0
    assert hybrid.monthly_physical["Purchased Cassava Delivered ton"].sum() > 0


def test_all_products_follow_the_staged_derivative_hierarchy(integrated_results):
    _, results = integrated_results
    output = results["HYBRID"]["integrated_cycle"]

    required_columns = [
        "Ethanol litres",
        "HQCF ton",
        "Garri ton",
        "Industrial Starch ton",
        "Dextrin ton",
        "Glucose Syrup ton",
        "Sorbitol ton",
        "Animal Feed ton",
    ]
    assert (output.product_monthly[required_columns].sum() > 0).all()

    ledger = output.processing_ledger
    starch = ledger.loc[ledger["Output Stream"] == "Starch Pool"]
    glucose_pool = ledger.loc[ledger["Output Stream"] == "Glucose Syrup Pool"]
    sorbitol = ledger.loc[ledger["Output Stream"] == "Sorbitol"]
    assert set(starch["Stage Order"]) == {1}
    assert set(glucose_pool["Stage Order"]) == {2}
    assert set(sorbitol["Stage Order"]) == {3}
    assert starch["Feedstock Grade"].str.contains("Large/Older", regex=False).all()
    assert (starch["Dry Matter Yield Factor"] > 1.0).all()

    food = ledger.loc[ledger["Output Stream"].isin(["HQCF", "Garri"])]
    assert food["Feedstock Grade"].str.contains("Fresh", regex=False).all()
    assert (food["Maximum Feedstock Age Days"] <= 2).all()


def test_mass_balance_transfer_pricing_and_eliminations_reconcile(integrated_results):
    _, results = integrated_results
    output = results["HYBRID"]["integrated_cycle"]

    balance_columns = [
        "Raw Cassava Balance Delta",
        "Starch Balance Delta",
        "Glucose Balance Delta",
        "Residue Balance Delta",
    ]
    assert np.allclose(output.mass_balance_monthly[balance_columns].to_numpy(), 0.0, atol=1e-6)
    assert output.metrics["Mass Balance Passed"] == 1.0
    assert np.allclose(
        output.eliminations_monthly["Net Intercompany Elimination"],
        0.0,
        atol=1e-6,
    )
    assert output.segment_monthly["FarmCo Revenue"].sum() == pytest.approx(
        output.segment_monthly["ProcessingCo Farm Transfer Cost"].sum()
    )
    assert output.farm_income_monthly["Farm Transfer Revenue"].sum() == pytest.approx(
        output.monthly_physical["Farm Transfer Revenue"].sum()
    )


def test_commercialization_caps_sales_and_builds_product_revenue(integrated_results):
    _, results = integrated_results
    output = results["HYBRID"]["integrated_cycle"]
    ledger = output.commercialization_ledger

    assert set(ledger["Product"]) == {
        "Fuel Ethanol",
        "HQCF",
        "Garri",
        "Industrial Starch",
        "Dextrin",
        "Glucose Syrup",
        "Sorbitol",
        "Animal Feed",
    }
    assert (ledger["Sales Volume"] <= ledger["Available for Sale"] + 1e-9).all()
    assert (ledger["Ending Inventory"] >= 0).all()
    revenue_columns = [
        column for column in output.revenue_monthly.columns if column != "Total Revenue"
    ]
    pd.testing.assert_series_equal(
        output.revenue_monthly["Total Revenue"],
        output.revenue_monthly[revenue_columns].sum(axis=1),
        check_names=False,
    )
    assert output.revenue_monthly["Total Revenue"].sum() > 0


def test_cycle_validation_rejects_invalid_crop_and_processing_durations():
    model = CassavaBioethanolModel.default()
    page = model.input_page
    cycle = page.annual_cycle_plan.data.copy()
    cycle.loc[:, "Cultivation Months"] = 8
    cycle.loc[:, "Processing Months"] = 4
    page.annual_cycle_plan.set_data(cycle, mark_user_input=True)

    with pytest.raises(ValueError, match="Cultivation Months"):
        model.build("HYBRID")


def test_new_input_tables_rehydrate_from_dict():
    page = default_input_page()
    rebuilt = InputLandingPage.from_dict(
        {
            "annual_cycle_plan": page.annual_cycle_plan.to_dict(),
            "farm_cost_assumptions": page.farm_cost_assumptions.to_dict(),
            "farm_capex": page.farm_capex.to_dict(),
            "procurement_plan": page.procurement_plan.to_dict(),
            "product_routing": page.product_routing.to_dict(),
            "commercialization_plan": page.commercialization_plan.to_dict(),
        }
    )

    assert rebuilt.annual_cycle_plan.data.shape == page.annual_cycle_plan.data.shape
    assert len(rebuilt.farm_cost_assumptions.data) >= 10
    assert {"Dextrin", "Glucose Syrup", "Sorbitol"}.issubset(
        set(rebuilt.commercialization_plan.data["Product"])
    )


def test_integrated_excel_pages_are_exported(tmp_path, integrated_results):
    _, results = integrated_results
    path = tmp_path / "integrated_cycle_pages.xlsx"
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        _write_integrated_cycle_pages(writer, results["HYBRID"])

    workbook = pd.ExcelFile(path)
    assert {
        "Cycle Model",
        "FarmCo Model",
        "Cassava Sourcing",
        "Product Routing",
        "Commercialisation",
        "Segments & Eliminations",
        "Mass Balance",
    }.issubset(set(workbook.sheet_names))
