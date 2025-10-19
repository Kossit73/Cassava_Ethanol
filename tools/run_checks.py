#!/usr/bin/env python3
"""Utility to compile the project and optionally run a quick scenario build."""
from __future__ import annotations

import compileall
import json
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


def _print_summary(summary_dict: dict, prefix: str = "") -> None:
    for scenario in summary_dict.get("scenarios", []):
        warnings = scenario.get("warnings") or []
        status = "ok" if not warnings else "warn"
        message = (
            f"[{prefix}{status}] {scenario.get('scenario', 'UNKNOWN')}: "
            f"passes={scenario.get('passes', 0)} "
            f"balance_gap={float(scenario.get('balance_gap', 0.0)):.6g} "
            f"cash_gap={float(scenario.get('cash_gap', 0.0)):.6g}"
        )
        print(message)
        for warning in warnings:
            print(f"    -> {warning}")


def _load_cached_summary() -> dict | None:
    baseline = ROOT / "tools" / "diagnostics_baseline.json"
    if not baseline.exists():
        return None
    try:
        with baseline.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:  # pragma: no cover - diagnostic convenience
        print(f"[skip] Failed to load cached diagnostics: {exc}", file=sys.stderr)
        return None


def _run_diagnostics() -> None:
    """Run recursive scenario diagnostics when dependencies are available."""

    try:
        from bioethanol_model import run_recursive_checks  # type: ignore
    except ModuleNotFoundError as exc:
        cached = _load_cached_summary()
        if cached:
            print(
                "[info] Dependencies missing; using cached diagnostics summary.",
                file=sys.stderr,
            )
            _print_summary(cached, prefix="cached-")
        else:
            print(f"[skip] Diagnostics skipped: missing dependency: {exc}", file=sys.stderr)
        return

    try:
        summary = run_recursive_checks(max_passes=3)
    except Exception as exc:  # pragma: no cover - surfaced in CLI output
        print(f"[fail] Diagnostics raised an error: {exc}", file=sys.stderr)
        raise

    summary_dict = summary.to_dict()
    _print_summary(summary_dict)


def _cleanup_pycache() -> None:
    for path in ROOT.rglob("__pycache__"):
        if path.is_dir() and ROOT in path.parents:
            shutil.rmtree(path)


def main() -> None:
    _compile_targets()
    _run_diagnostics()
    _cleanup_pycache()


if __name__ == "__main__":
    main()
