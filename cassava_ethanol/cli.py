"""Command line interface for the cassava ethanol financial model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml

from . import (
    CassavaEthanolModel,
    ModelInputs,
    Scenario,
    ScenarioRunner,
    format_cash_flow_table,
    format_summary,
)
from .inputs import (
    CapitalPlan,
    FeedstockAssumptions,
    FinancialAssumptions,
    OperatingCosts,
    PlantProfile,
    ProductPricing,
)


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cassava ethanol financial planning toolkit"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/base_config.yaml"),
        help="Path to a YAML configuration file",
    )
    parser.add_argument(
        "--scenarios",
        type=Path,
        help="Optional YAML file describing scenario overrides",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Return results as JSON instead of formatted text",
    )
    return parser.parse_args(args)


def load_inputs(config_path: Path) -> ModelInputs:
    data = yaml.safe_load(config_path.read_text())
    plant = PlantProfile(**data["plant"])
    feedstock = FeedstockAssumptions(**data["feedstock"])
    pricing = ProductPricing(**data["pricing"])
    operating = OperatingCosts(**data["operating_costs"])
    capital = CapitalPlan(**data["capital"])
    financial = FinancialAssumptions(**data["financial"])
    return ModelInputs(
        plant=plant,
        feedstock=feedstock,
        pricing=pricing,
        operating_costs=operating,
        capital=capital,
        financial=financial,
    )


def load_scenarios(path: Path | None) -> Iterable[Scenario]:
    if path is None:
        return []
    data = yaml.safe_load(path.read_text())
    scenarios = []
    for item in data:
        scenarios.append(Scenario(name=item["name"], overrides=item["overrides"]))
    return scenarios


def serialize_results(result) -> Dict[str, Any]:
    return {
        "npv": result.npv,
        "irr": result.irr,
        "payback_year": result.payback_year,
        "cash_flows": [
            {
                "year": row.year,
                "production_liters": row.production_liters,
                "revenue": row.revenue,
                "feedstock_cost": row.feedstock_cost,
                "variable_operating_cost": row.variable_operating_cost,
                "fixed_operating_cost": row.fixed_operating_cost,
                "maintenance_cost": row.maintenance_cost,
                "ebitda": row.ebitda,
                "depreciation": row.depreciation,
                "ebit": row.ebit,
                "tax": row.tax,
                "net_income": row.net_income,
                "working_capital_change": row.working_capital_change,
                "capital_expenditure": row.capital_expenditure,
                "free_cash_flow": row.free_cash_flow,
                "discounted_cash_flow": row.discounted_cash_flow,
                "cumulative_cash_flow": row.cumulative_cash_flow,
            }
            for row in result.cash_flows
        ],
    }


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    inputs = load_inputs(args.config)
    model = CassavaEthanolModel(inputs)
    result = model.run()

    scenarios = list(load_scenarios(args.scenarios))
    if args.json:
        payload = {
            "base": serialize_results(result),
        }
        if scenarios:
            runner = ScenarioRunner(inputs, scenarios)
            payload["scenarios"] = [
                {
                    "name": scenario_result.scenario.name,
                    "results": serialize_results(scenario_result.results),
                }
                for scenario_result in runner.run()
            ]
        print(json.dumps(payload, indent=2))
        return 0

    print(format_summary(result))
    print()
    print(format_cash_flow_table(result.cash_flows))

    if scenarios:
        print("\nScenario comparison:")
        runner = ScenarioRunner(inputs, scenarios)
        for scenario_result in runner.run():
            print("\n" + scenario_result.scenario.name)
            print(format_summary(scenario_result.results))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
