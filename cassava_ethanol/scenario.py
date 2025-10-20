"""Scenario planning utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

from .inputs import ModelInputs
from .model import CassavaEthanolModel, ModelResults


@dataclass(slots=True)
class Scenario:
    """Definition of a single modelling scenario."""

    name: str
    overrides: Dict[str, float]

    def apply(self, inputs: ModelInputs) -> ModelInputs:
        return inputs.copy_with_overrides(self.overrides)


@dataclass(slots=True)
class ScenarioResult:
    """Pairing of scenario metadata and run results."""

    scenario: Scenario
    results: ModelResults


class ScenarioRunner:
    """Execute a suite of scenarios for comparison."""

    def __init__(self, base_inputs: ModelInputs, scenarios: Iterable[Scenario]):
        self.base_inputs = base_inputs
        self.scenarios = list(scenarios)

    def run(self) -> List[ScenarioResult]:
        output: List[ScenarioResult] = []
        for scenario in self.scenarios:
            model = CassavaEthanolModel(scenario.apply(self.base_inputs))
            output.append(ScenarioResult(scenario=scenario, results=model.run()))
        return output


__all__ = ["Scenario", "ScenarioResult", "ScenarioRunner"]
