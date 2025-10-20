from __future__ import annotations

import argparse
from pathlib import Path

# The heavy modelling stack depends on pandas/numpy/xlsxwriter.  Importing the
# modules lazily inside ``run_standard_export`` lets us detect missing optional
# dependencies and fall back to a lightweight workbook generator when the
# packages are not available (common in the execution environment used for the
# kata).

DEFAULT_SCENARIOS = ("FARM_ONLY", "BUY_ONLY", "HYBRID")


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
        choices=DEFAULT_SCENARIOS,
        default="FARM_ONLY",
        help="Scenario to run when exporting a single workbook.",
    )
    parser.add_argument(
        "--all-scenarios",
        action="store_true",
        help="Generate a separate workbook for each scenario (FARM_ONLY, BUY_ONLY, HYBRID).",
    )
    return parser.parse_args()


def run_standard_export(args: argparse.Namespace) -> None:
    from bioethanol_model import CassavaBioethanolModel
    from bioethanol_model.exporter import export_to_excel
    from bioethanol_model.inputs import default_input_page
    from bioethanol_model.sensitivity import SensitivityScenario
    from bioethanol_model.scenario import ScenarioConfig

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


def run_fallback_export(args: argparse.Namespace, missing_dependency: str) -> None:
    from fallback_model import CassavaFallbackModel

    model = CassavaFallbackModel()

    if args.all_scenarios:
        base_path = args.output
        parent = base_path.parent or Path.cwd()
        stem = base_path.stem if base_path.suffix else base_path.name
        suffix = base_path.suffix if base_path.suffix else ".xlsx"
        for scenario_name in model.scenarios:
            workbook_path = parent / f"{stem}_{scenario_name}{suffix}"
            model.export(workbook_path, scenario=scenario_name)
            print(
                "Simplified financial model saved to"
                f" {workbook_path} (fallback path used because {missing_dependency} is unavailable)."
            )
    else:
        scenario_name = args.scenario
        model.export(args.output, scenario=scenario_name)
        print(
            f"Simplified financial model saved to {args.output} for scenario {scenario_name}"
            f" (fallback path used because {missing_dependency} is unavailable)."
        )


def main() -> None:
    args = parse_args()

    try:
        run_standard_export(args)
    except ModuleNotFoundError as exc:
        missing = exc.name or "a required dependency"
        if missing.lower() in {"numpy", "pandas", "xlsxwriter"}:
            run_fallback_export(args, missing)
        else:  # pragma: no cover - re-raise unexpected import errors
            raise


if __name__ == "__main__":
    main()
