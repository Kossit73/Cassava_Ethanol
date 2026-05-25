"""Cassava bioethanol financial modeling toolkit."""

from importlib import import_module
from typing import Any

from .financial_model import CassavaBioethanolModel
from .diagnostics import DiagnosticSummary, ScenarioDiagnostics, run_recursive_checks

__all__ = [
    "CassavaBioethanolModel",
    "AdvancedAnalyticsToolkit",
    "DiagnosticSummary",
    "ScenarioDiagnostics",
    "run_recursive_checks",
]


def __getattr__(name: str) -> Any:
    if name == "AdvancedAnalyticsToolkit":
        module = import_module(".advanced_tools", __name__)
        return module.AdvancedAnalyticsToolkit
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
