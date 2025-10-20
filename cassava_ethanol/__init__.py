"""Cassava Ethanol financial modeling toolkit.

This package provides data models and utilities for evaluating
cassava-based ethanol production projects.  The primary entry point is
:class:`cassava_ethanol.model.CassavaEthanolModel` which orchestrates the
cash-flow calculations and financial metrics that matter most when
assessing new greenfield opportunities.
"""

from .inputs import (
    CapitalPlan,
    FeedstockAssumptions,
    FinancialAssumptions,
    ModelInputs,
    OperatingCosts,
    PlantProfile,
    ProductPricing,
)
from .model import CassavaEthanolModel, CashFlowBreakdown, ModelResults
from .scenario import Scenario, ScenarioResult, ScenarioRunner
from .report import format_cash_flow_table, format_summary

__all__ = [
    "CapitalPlan",
    "FeedstockAssumptions",
    "FinancialAssumptions",
    "ModelInputs",
    "OperatingCosts",
    "PlantProfile",
    "ProductPricing",
    "CassavaEthanolModel",
    "CashFlowBreakdown",
    "ModelResults",
    "Scenario",
    "ScenarioResult",
    "ScenarioRunner",
    "format_cash_flow_table",
    "format_summary",
]
