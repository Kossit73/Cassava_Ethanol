"""Formatting helpers for presenting model results."""

from __future__ import annotations

from typing import Iterable, Sequence

from .model import CashFlowBreakdown, ModelResults


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def format_volume(value: float) -> str:
    return f"{value:,.0f} L"


def format_percentage(value: float) -> str:
    return f"{value * 100:.1f}%"


def format_cash_flow_table(rows: Sequence[CashFlowBreakdown]) -> str:
    """Return a human readable table summarizing annual cash flows."""

    headers = [
        "Year",
        "Production",
        "Revenue",
        "Feedstock",
        "Variable Opex",
        "Fixed Opex",
        "Maintenance",
        "EBITDA",
        "Free Cash Flow",
        "Cumulative",
    ]

    lines = [" | ".join(headers), " | ".join("-" * len(h) for h in headers)]
    for row in rows:
        lines.append(
            " | ".join(
                [
                    str(row.year),
                    format_volume(row.production_liters),
                    format_currency(row.revenue),
                    format_currency(row.feedstock_cost),
                    format_currency(row.variable_operating_cost),
                    format_currency(row.fixed_operating_cost),
                    format_currency(row.maintenance_cost),
                    format_currency(row.ebitda),
                    format_currency(row.free_cash_flow),
                    format_currency(row.cumulative_cash_flow),
                ]
            )
        )
    return "\n".join(lines)


def format_summary(results: ModelResults) -> str:
    """Return a one-page summary of the model outputs."""

    buffer = [
        f"Project: {results.inputs.plant.name}",
        f"NPV: {format_currency(results.npv)}",
        f"IRR: {format_percentage(results.irr) if results.irr == results.irr else 'n/a'}",
    ]
    if results.payback_year is not None:
        buffer.append(f"Payback: {results.payback_year:.1f} years")
    else:
        buffer.append("Payback: beyond analysis horizon")
    return "\n".join(buffer)


__all__ = [
    "format_cash_flow_table",
    "format_currency",
    "format_percentage",
    "format_summary",
    "format_volume",
]
