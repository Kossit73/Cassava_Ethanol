"""Regression-style tests for the cassava ethanol model."""

from cassava_ethanol import (
    CapitalPlan,
    CassavaEthanolModel,
    FeedstockAssumptions,
    FinancialAssumptions,
    ModelInputs,
    OperatingCosts,
    PlantProfile,
    ProductPricing,
)


def build_default_inputs() -> ModelInputs:
    plant = PlantProfile(
        name="Unit Test Plant",
        capacity_liters_per_day=12000,
        operating_days_per_year=330,
        ramp_up_profile=(0.6, 0.85, 1.0),
    )
    feedstock = FeedstockAssumptions(
        cassava_price_per_ton=55,
        cassava_required_per_liter_kg=2.8,
        transport_cost_per_ton=8,
        price_escalation=0.01,
    )
    pricing = ProductPricing(
        ethanol_price_per_liter=1.15,
        price_escalation=0.015,
        ddgs_price_per_ton=140,
        ddgs_output_per_liter_kg=0.22,
    )
    operating = OperatingCosts(
        fixed_operating_cost=1_050_000,
        variable_operating_cost_per_liter=0.09,
        maintenance_cost_percent_of_capex=0.025,
    )
    capital = CapitalPlan(
        initial_investment=6_000_000,
        working_capital_percent_of_revenue=0.07,
        depreciation_years=8,
        salvage_value_percent=0.1,
        tax_rate=0.25,
    )
    financial = FinancialAssumptions(
        discount_rate=0.11,
        inflation_rate=0.02,
        analysis_years=10,
    )
    return ModelInputs(
        plant=plant,
        feedstock=feedstock,
        pricing=pricing,
        operating_costs=operating,
        capital=capital,
        financial=financial,
    )


def test_model_produces_positive_npv():
    model = CassavaEthanolModel(build_default_inputs())
    results = model.run()
    assert results.npv > 0
    assert 0 < results.irr < 1
    assert results.payback_year is not None


def test_scenario_override_updates_inputs():
    inputs = build_default_inputs()
    updated = inputs.copy_with_overrides({"pricing.ethanol_price_per_liter": 1.2})
    assert updated.pricing.ethanol_price_per_liter == 1.2
    assert inputs.pricing.ethanol_price_per_liter == 1.15
