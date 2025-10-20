"""Core financial model for cassava ethanol projects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from .inputs import ModelInputs


@dataclass(slots=True)
class CashFlowBreakdown:
    """Detailed result for a single model period."""

    year: int
    production_liters: float
    revenue: float
    feedstock_cost: float
    variable_operating_cost: float
    fixed_operating_cost: float
    maintenance_cost: float
    ebitda: float
    depreciation: float
    ebit: float
    tax: float
    net_income: float
    working_capital_change: float
    capital_expenditure: float
    free_cash_flow: float
    discounted_cash_flow: float
    cumulative_cash_flow: float


@dataclass(slots=True)
class ModelResults:
    """Aggregate financial results."""

    inputs: ModelInputs
    cash_flows: List[CashFlowBreakdown]
    npv: float
    irr: float
    payback_year: float | None

    def cash_flow_series(self) -> List[float]:
        return [row.free_cash_flow for row in self.cash_flows]


class CassavaEthanolModel:
    """Financial model covering production, revenue and cash-flow analysis."""

    def __init__(self, inputs: ModelInputs):
        self.inputs = inputs

    def run(self) -> ModelResults:
        cash_flows = []
        cumulative = 0.0
        discount_rate = self.inputs.financial.discount_rate

        # Initial investment at year 0
        initial_cf = -self.inputs.capital.initial_investment
        discounted_initial = initial_cf
        cumulative += initial_cf
        cash_flows.append(
            CashFlowBreakdown(
                year=0,
                production_liters=0.0,
                revenue=0.0,
                feedstock_cost=0.0,
                variable_operating_cost=0.0,
                fixed_operating_cost=0.0,
                maintenance_cost=0.0,
                ebitda=0.0,
                depreciation=0.0,
                ebit=0.0,
                tax=0.0,
                net_income=0.0,
                working_capital_change=0.0,
                capital_expenditure=self.inputs.capital.initial_investment,
                free_cash_flow=initial_cf,
                discounted_cash_flow=discounted_initial,
                cumulative_cash_flow=cumulative,
            )
        )

        working_capital_previous = 0.0
        npv = discounted_initial
        cash_flow_series = [initial_cf]
        inflation = self.inputs.financial.inflation_rate
        initial_capex = self.inputs.capital.initial_investment

        for year in range(1, self.inputs.financial.analysis_years + 1):
            production = self.inputs.plant.production_volume(year)
            revenue = self._compute_revenue(year, production)
            feedstock_cost = self._compute_feedstock_cost(year, production)
            variable_cost = (
                self.inputs.operating_costs.variable_cost_for_year(year, inflation)
                * production
            )
            fixed_cost = self.inputs.operating_costs.fixed_cost_for_year(year, inflation)
            maintenance_cost = self.inputs.operating_costs.maintenance_cost_for_year(
                year, inflation, initial_capex
            )
            ebitda = (
                revenue
                - feedstock_cost
                - variable_cost
                - fixed_cost
                - maintenance_cost
            )
            depreciation = self._depreciation_for_year(year)
            ebit = ebitda - depreciation
            tax = max(ebit, 0.0) * self.inputs.capital.tax_rate
            net_income = ebit - tax

            target_working_capital = (
                revenue * self.inputs.capital.working_capital_percent_of_revenue
            )
            working_capital_change = target_working_capital - working_capital_previous
            working_capital_previous = target_working_capital

            free_cash_flow = net_income + depreciation - working_capital_change

            if year == self.inputs.financial.analysis_years:
                salvage = (
                    self.inputs.capital.initial_investment
                    * self.inputs.capital.salvage_value_percent
                )
                free_cash_flow += salvage + working_capital_previous
                working_capital_change -= working_capital_previous
                working_capital_previous = 0.0

            discount_factor = (1 + discount_rate) ** year
            discounted = free_cash_flow / discount_factor
            cumulative += free_cash_flow
            npv += discounted
            cash_flow_series.append(free_cash_flow)

            cash_flows.append(
                CashFlowBreakdown(
                    year=year,
                    production_liters=production,
                    revenue=revenue,
                    feedstock_cost=feedstock_cost,
                    variable_operating_cost=variable_cost,
                    fixed_operating_cost=fixed_cost,
                    maintenance_cost=maintenance_cost,
                    ebitda=ebitda,
                    depreciation=depreciation,
                    ebit=ebit,
                    tax=tax,
                    net_income=net_income,
                    working_capital_change=working_capital_change,
                    capital_expenditure=0.0,
                    free_cash_flow=free_cash_flow,
                    discounted_cash_flow=discounted,
                    cumulative_cash_flow=cumulative,
                )
            )

        irr = _calculate_irr(cash_flow_series)
        payback = _calculate_payback(cash_flow_series)

        return ModelResults(
            inputs=self.inputs,
            cash_flows=cash_flows,
            npv=npv,
            irr=irr,
            payback_year=payback,
        )

    def _compute_revenue(self, year: int, production_liters: float) -> float:
        pricing = self.inputs.pricing
        ethanol_price = pricing.ethanol_price_for_year(year)
        revenue = production_liters * ethanol_price

        if pricing.ddgs_output_per_liter_kg and pricing.ddgs_price_per_ton:
            ddgs_tons = (
                production_liters * pricing.ddgs_output_per_liter_kg / 1000.0
            )
            revenue += ddgs_tons * pricing.ddgs_price_for_year(year)

        if pricing.co2_output_per_liter_kg and pricing.co2_price_per_ton:
            co2_tons = production_liters * pricing.co2_output_per_liter_kg / 1000.0
            revenue += co2_tons * pricing.co2_price_for_year(year)

        return revenue

    def _compute_feedstock_cost(self, year: int, production_liters: float) -> float:
        feedstock = self.inputs.feedstock
        cassava_price = feedstock.price_for_year(year)
        cassava_tons = (
            production_liters * feedstock.cassava_required_per_liter_kg / 1000.0
        )
        transport_cost = cassava_tons * feedstock.transport_cost_per_ton
        return cassava_tons * cassava_price + transport_cost

    def _depreciation_for_year(self, year: int) -> float:
        depreciation_years = self.inputs.capital.depreciation_years
        if year > depreciation_years:
            return 0.0
        return self.inputs.capital.initial_investment / depreciation_years


def _calculate_irr(cash_flows: Iterable[float]) -> float:
    flows = list(cash_flows)
    if not flows:
        return float("nan")
    positive = any(cf > 0 for cf in flows)
    negative = any(cf < 0 for cf in flows)
    if not (positive and negative):
        return float("nan")

    def npv(rate: float) -> float:
        return sum(cf / (1 + rate) ** i for i, cf in enumerate(flows))

    def derivative(rate: float) -> float:
        return sum(-i * cf / (1 + rate) ** (i + 1) for i, cf in enumerate(flows) if i)

    guess = 0.1
    for _ in range(100):
        value = npv(guess)
        if abs(value) < 1e-6:
            return float(guess)
        slope = derivative(guess)
        if slope == 0:
            break
        guess -= value / slope
        if guess < -0.99:
            guess = -0.99
    return float("nan")


def _calculate_payback(cash_flows: List[float]) -> float | None:
    cumulative = 0.0
    for year, cf in enumerate(cash_flows):
        cumulative += cf
        if cumulative >= 0:
            if year == 0:
                return 0.0
            previous_cumulative = cumulative - cf
            if cf == 0:
                return float(year)
            fraction = (0 - previous_cumulative) / cf
            return year - 1 + fraction
    return None


__all__ = [
    "CashFlowBreakdown",
    "ModelResults",
    "CassavaEthanolModel",
]
