"""Structured input data definitions for the cassava ethanol model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Sequence


@dataclass(slots=True)
class PlantProfile:
    """Production characteristics of the ethanol plant."""

    name: str
    capacity_liters_per_day: float
    operating_days_per_year: int
    ramp_up_profile: Sequence[float] = field(
        default_factory=lambda: (0.6, 0.8, 1.0)
    )

    def annual_capacity(self) -> float:
        return self.capacity_liters_per_day * self.operating_days_per_year

    def production_volume(self, year: int) -> float:
        """Return the expected production volume for ``year`` (1-indexed)."""

        if year <= 0:
            raise ValueError("Year must be greater than zero")
        if year <= len(self.ramp_up_profile):
            factor = self.ramp_up_profile[year - 1]
        else:
            factor = self.ramp_up_profile[-1]
        return self.annual_capacity() * factor


@dataclass(slots=True)
class FeedstockAssumptions:
    """Parameters describing cassava sourcing."""

    cassava_price_per_ton: float
    cassava_required_per_liter_kg: float
    transport_cost_per_ton: float = 0.0
    price_escalation: float = 0.0

    def price_for_year(self, year: int) -> float:
        if year < 1:
            raise ValueError("Year index must be >= 1")
        return self.cassava_price_per_ton * (1 + self.price_escalation) ** (year - 1)


@dataclass(slots=True)
class ProductPricing:
    """Revenue drivers for ethanol and co-products."""

    ethanol_price_per_liter: float
    price_escalation: float = 0.0
    ddgs_price_per_ton: float = 0.0
    ddgs_output_per_liter_kg: float = 0.0
    co2_price_per_ton: float = 0.0
    co2_output_per_liter_kg: float = 0.0

    def ethanol_price_for_year(self, year: int) -> float:
        return self.ethanol_price_per_liter * (1 + self.price_escalation) ** (year - 1)

    def ddgs_price_for_year(self, year: int) -> float:
        return self.ddgs_price_per_ton * (1 + self.price_escalation) ** (year - 1)

    def co2_price_for_year(self, year: int) -> float:
        return self.co2_price_per_ton * (1 + self.price_escalation) ** (year - 1)


@dataclass(slots=True)
class OperatingCosts:
    """Operating expenses other than feedstock."""

    fixed_operating_cost: float
    variable_operating_cost_per_liter: float
    maintenance_cost_percent_of_capex: float

    def fixed_cost_for_year(self, year: int, inflation_rate: float) -> float:
        return self.fixed_operating_cost * (1 + inflation_rate) ** (year - 1)

    def variable_cost_for_year(
        self, year: int, inflation_rate: float
    ) -> float:
        return self.variable_operating_cost_per_liter * (1 + inflation_rate) ** (year - 1)

    def maintenance_cost_for_year(
        self, year: int, inflation_rate: float, initial_capex: float
    ) -> float:
        return (
            initial_capex
            * self.maintenance_cost_percent_of_capex
            * (1 + inflation_rate) ** (year - 1)
        )


@dataclass(slots=True)
class CapitalPlan:
    """Capital expenditure and corporate finance assumptions."""

    initial_investment: float
    working_capital_percent_of_revenue: float
    depreciation_years: int
    salvage_value_percent: float
    tax_rate: float


@dataclass(slots=True)
class FinancialAssumptions:
    """Financial evaluation controls."""

    discount_rate: float
    inflation_rate: float
    analysis_years: int


@dataclass(slots=True)
class ModelInputs:
    """Container grouping all model input structures."""

    plant: PlantProfile
    feedstock: FeedstockAssumptions
    pricing: ProductPricing
    operating_costs: OperatingCosts
    capital: CapitalPlan
    financial: FinancialAssumptions

    def copy_with_overrides(self, overrides: Dict[str, float]) -> "ModelInputs":
        """Return a copy of the inputs with dotted-path overrides applied."""

        data = {
            "plant": self.plant,
            "feedstock": self.feedstock,
            "pricing": self.pricing,
            "operating_costs": self.operating_costs,
            "capital": self.capital,
            "financial": self.financial,
        }
        updated: Dict[str, object] = {}
        for key, value in data.items():
            updated[key] = dataclass_replace(value)

        for dotted_path, new_value in overrides.items():
            segments = dotted_path.split(".")
            if len(segments) != 2:
                raise ValueError(
                    f"Invalid override '{dotted_path}'. Expected format 'section.field'."
                )
            section, field_name = segments
            if section not in updated:
                raise KeyError(f"Unknown section '{section}' in override '{dotted_path}'.")
            section_obj = updated[section]
            if not hasattr(section_obj, field_name):
                raise KeyError(
                    f"Unknown field '{field_name}' in section '{section}'."
                )
            setattr(section_obj, field_name, new_value)

        return ModelInputs(
            plant=updated["plant"],
            feedstock=updated["feedstock"],
            pricing=updated["pricing"],
            operating_costs=updated["operating_costs"],
            capital=updated["capital"],
            financial=updated["financial"],
        )


def dataclass_replace(instance):
    """Return a shallow copy of a dataclass instance."""

    cls = type(instance)
    if not hasattr(cls, "__dataclass_fields__"):
        raise TypeError("dataclass_replace expects a dataclass instance")
    values = {field: getattr(instance, field) for field in instance.__dataclass_fields__}
    return cls(**values)


def iter_dotted_paths(obj: object) -> Iterable[str]:
    """Yield dotted paths for all dataclass fields.

    This utility helps users discover valid override keys.
    """

    if not hasattr(obj, "__dataclass_fields__"):
        return
    for field_name, field_info in obj.__dataclass_fields__.items():
        value = getattr(obj, field_name)
        yield field_name
        if hasattr(value, "__dataclass_fields__"):
            for nested in iter_dotted_paths(value):
                yield f"{field_name}.{nested}"


__all__ = [
    "PlantProfile",
    "FeedstockAssumptions",
    "ProductPricing",
    "OperatingCosts",
    "CapitalPlan",
    "FinancialAssumptions",
    "ModelInputs",
    "dataclass_replace",
    "iter_dotted_paths",
]
