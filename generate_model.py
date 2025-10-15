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
        help="Path for the generated Excel workbook.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = CassavaBioethanolModel(default_input_page())

    sensitivity = [
        SensitivityScenario("Tax rate +1%", "Corporate tax rate", 0.01),
        SensitivityScenario("Tax rate -1%", "Corporate tax rate", -0.01),
    ]

    scenarios = [
        ScenarioConfig("High price", {"Investor share capital": 0.5}),
        ScenarioConfig("Low price", {"Investor share capital": 0.4}),
    ]

    export_to_excel(model, args.output, sensitivity_scenarios=sensitivity, scenario_configs=scenarios)
    print(f"Financial model saved to {args.output}")


if __name__ == "__main__":
    main()
