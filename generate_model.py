from __future__ import annotations

import argparse
from pathlib import Path

from bioethanol_model import CassavaBioethanolModel
from bioethanol_model.exporter import export_to_excel
from bioethanol_model.inputs import default_input_page
from bioethanol_model.sensitivity import SensitivityScenario
from bioethanol_model.scenario import ScenarioConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Cassava Bioethanol financial model workbook.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Cassava_Bioethanol_Financial_Model.xlsx"),
        help="Path for the generated Excel workbook or base filename when exporting all scenarios.",
    )
    parser.add_argument(
        "--scenario",
        choices=CassavaBioethanolModel.SCENARIOS,
        default="FARM_ONLY",
        help="Scenario to run when exporting a single workbook.",
    )
    parser.add_argument(
        "--all-scenarios",
        action="store_true",
        help="Generate a separate workbook for each scenario (FARM_ONLY, BUY_ONLY, HYBRID).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sensitivity = [
        SensitivityScenario("Tax rate +1%", "Corporate tax rate", 0.01),
        SensitivityScenario("Tax rate -1%", "Corporate tax rate", -0.01),
    ]

    scenarios = [
        ScenarioConfig("High price", {"Investor share capital": 0.5}),
        ScenarioConfig("Low price", {"Investor share capital": 0.4}),
    ]

    if args.all_scenarios:
        base_path = args.output
        parent = base_path.parent or Path.cwd()
        stem = base_path.stem if base_path.suffix else base_path.name
        suffix = base_path.suffix if base_path.suffix else ".xlsx"
        for scenario_name in CassavaBioethanolModel.SCENARIOS:
            model = CassavaBioethanolModel(default_input_page())
            workbook_path = parent / f"{stem}_{scenario_name}{suffix}"
            export_to_excel(
                model,
                workbook_path,
                sensitivity_scenarios=sensitivity,
                scenario_configs=scenarios,
                scenario=scenario_name,
            )
            print(f"Financial model saved to {workbook_path}")
    else:
        scenario_name = args.scenario
        model = CassavaBioethanolModel(default_input_page())
        export_to_excel(
            model,
            args.output,
            sensitivity_scenarios=sensitivity,
            scenario_configs=scenarios,
            scenario=scenario_name,
        )
        print(f"Financial model saved to {args.output} for scenario {scenario_name}")


if __name__ == "__main__":
    main()
