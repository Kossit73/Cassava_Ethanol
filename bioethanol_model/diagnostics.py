"""Diagnostics utilities for validating model outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional

import pandas as pd

from .financial_model import CassavaBioethanolModel


@dataclass
class ScenarioDiagnostics:
    """Summary of validation checks for a single scenario."""

    scenario: str
    passes: int
    balance_gap: float
    cash_gap: float
    warnings: List[str] = field(default_factory=list)


@dataclass
class DiagnosticSummary:
    """Aggregated diagnostics for one input snapshot."""

    signature: str
    scenarios: List[ScenarioDiagnostics]

    def has_failures(self, tolerance: float = 1e-6) -> bool:
        return any(
            diag.balance_gap > tolerance or diag.cash_gap > tolerance
            for diag in self.scenarios
        )

    def to_dict(self) -> dict:
        return {
            "signature": self.signature,
            "scenarios": [
                {
                    "scenario": diag.scenario,
                    "passes": diag.passes,
                    "balance_gap": diag.balance_gap,
                    "cash_gap": diag.cash_gap,
                    "warnings": list(diag.warnings),
                }
                for diag in self.scenarios
            ],
        }


def _max_balance_gap(financials) -> float:
    balance = getattr(financials, "balance_monthly", pd.DataFrame())
    if balance.empty:
        return 0.0
    lhs = balance.get("Total Assets")
    rhs = balance.get("Total Liabilities & Equity")
    if lhs is None or rhs is None:
        return 0.0
    gap = (lhs - rhs).abs()
    return float(gap.max()) if not gap.empty else 0.0


def _cash_reconciliation_gap(financials) -> float:
    balance = getattr(financials, "balance_monthly", pd.DataFrame())
    cash_series = balance.get("Cash")
    cashflow = getattr(financials, "cashflow_monthly", pd.DataFrame())
    net_cash = cashflow.get("Net Cash Flow")
    if cash_series is None or net_cash is None or cash_series.empty:
        return 0.0
    reconciliation = cash_series - net_cash.cumsum()
    reconciliation = reconciliation.abs().fillna(0.0)
    return float(reconciliation.max()) if not reconciliation.empty else 0.0


def _nan_warning(df: Optional[pd.DataFrame]) -> Optional[str]:
    if df is None or df.empty:
        return None
    if df.isna().values.any():
        return "Detected NaN values in computed schedules"
    return None


def run_recursive_checks(
    model: Optional[CassavaBioethanolModel] = None,
    scenarios: Optional[Iterable[str]] = None,
    max_passes: int = 3,
    tolerance: float = 1e-6,
) -> DiagnosticSummary:
    """Build each scenario repeatedly until outputs stabilise and validate them."""

    model = model or CassavaBioethanolModel()
    scenario_list = [s.upper() for s in (scenarios or model.SCENARIOS)]
    diagnostics: List[ScenarioDiagnostics] = []

    for scenario in scenario_list:
        last_signature: Optional[str] = None
        passes = 0
        result = None
        for iteration in range(1, max_passes + 1):
            passes = iteration
            result = model.build(scenario)
            current_signature = model.result_signature(result)
            if last_signature is not None and current_signature == last_signature:
                break
            last_signature = current_signature
        if result is None:
            continue
        financials = result.get("financials")
        balance_gap = _max_balance_gap(financials)
        cash_gap = _cash_reconciliation_gap(financials)
        warnings: List[str] = []
        if balance_gap > tolerance:
            warnings.append(f"Balance sheet gap exceeds tolerance ({balance_gap:.6f})")
        if cash_gap > tolerance:
            warnings.append(f"Cash reconciliation gap exceeds tolerance ({cash_gap:.6f})")
        for frame in (
            getattr(financials, "income_monthly", None),
            getattr(financials, "cashflow_monthly", None),
            getattr(financials, "balance_monthly", None),
        ):
            warning = _nan_warning(frame)
            if warning and warning not in warnings:
                warnings.append(warning)
        diagnostics.append(
            ScenarioDiagnostics(
                scenario=scenario,
                passes=passes,
                balance_gap=balance_gap,
                cash_gap=cash_gap,
                warnings=warnings,
            )
        )

    return DiagnosticSummary(signature=model.input_signature(), scenarios=diagnostics)


__all__ = [
    "DiagnosticSummary",
    "ScenarioDiagnostics",
    "run_recursive_checks",
]
