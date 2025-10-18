#!/usr/bin/env python3
"""Utility to compile the project and optionally run a quick scenario build."""
from __future__ import annotations

import compileall
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _compile_targets() -> None:
    """Byte-compile core modules to catch syntax issues."""
    compileall.compile_dir(str(ROOT / "bioethanol_model"), quiet=1)
    for target in ("streamlit_app.py", "generate_model.py"):
        compileall.compile_file(str(ROOT / target), quiet=1)


def _attempt_scenario_build() -> None:
    """Instantiate the default model and build the FARM_ONLY scenario.

    Missing optional dependencies (for example numpy or pandas) are reported
    without failing the check so environments without wheels can still run
    the compilation step.
    """
    try:
        from bioethanol_model import CassavaBioethanolModel  # type: ignore
    except ModuleNotFoundError as exc:
        print(f"[skip] Scenario build skipped: missing dependency: {exc}", file=sys.stderr)
        return

    model = CassavaBioethanolModel()
    try:
        model.build("FARM_ONLY")
    except Exception as exc:  # pragma: no cover - surfaced in CLI output
        print(f"[fail] Scenario build raised an error: {exc}", file=sys.stderr)
        raise
    else:
        print("[ok] Scenario build completed for FARM_ONLY")


def _cleanup_pycache() -> None:
    for path in ROOT.rglob("__pycache__"):
        if path.is_dir() and ROOT in path.parents:
            shutil.rmtree(path)


def main() -> None:
    _compile_targets()
    _attempt_scenario_build()
    _cleanup_pycache()


if __name__ == "__main__":
    main()
