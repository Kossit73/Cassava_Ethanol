from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional

import hashlib
import pandas as pd


@dataclass
class EditableTable:
    """Generic structure that supports row add/remove operations."""

    name: str
    columns: List[str]
    data: pd.DataFrame = field(default_factory=pd.DataFrame)
    placeholder: bool = False

    def __post_init__(self) -> None:
        self._set_data(self.data)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _coerce_dataframe(self, df: pd.DataFrame | None) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=self.columns)
        coerced = df.copy()
        for column in self.columns:
            if column not in coerced.columns:
                coerced[column] = None
        return coerced[self.columns]

    def _set_data(self, df: pd.DataFrame | None) -> None:
        self.data = self._coerce_dataframe(df)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_data(self, df: pd.DataFrame | None, *, mark_user_input: bool | None = None) -> None:
        """Replace the table contents with *df* and update the placeholder flag.

        Parameters
        ----------
        df:
            The dataframe to store. Columns not present in ``self.columns`` are
            ignored; missing columns are added with ``None`` values.
        mark_user_input:
            When ``True`` the table is flagged as containing user-provided data
            (placeholders are disabled). ``False`` keeps the existing
            placeholder flag, and ``None`` leaves the flag unchanged.
        """

        self._set_data(df)
        if mark_user_input is True:
            self.placeholder = False
        elif mark_user_input is False:
            self.placeholder = self.placeholder

    def mark_placeholder(self, value: bool) -> None:
        self.placeholder = bool(value)

    @property
    def model_frame(self) -> pd.DataFrame:
        """Return the dataframe used for calculations (empty when placeholder)."""

        if self.placeholder:
            return pd.DataFrame(columns=self.columns)
        return self.data.copy()

    def add_row(self, values: Dict[str, object]) -> None:
        missing = [c for c in self.columns if c not in values]
        if missing:
            raise ValueError(f"Missing values for columns: {missing}")
        self.data = pd.concat([self.data, pd.DataFrame([values])], ignore_index=True)
        self.placeholder = False

    def remove_row(self, index: int) -> None:
        if index not in self.data.index:
            raise KeyError(f"Row {index} not found in {self.name}")
        self.data = self.data.drop(index).reset_index(drop=True)
        self.placeholder = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "columns": self.columns,
            "data": self.data.copy(),
            "placeholder": self.placeholder,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EditableTable":
        data = payload.get("data")
        if isinstance(data, pd.DataFrame):
            frame = data.copy()
        elif data is None:
            frame = pd.DataFrame()
        else:
            frame = pd.DataFrame(data)
        return cls(
            name=str(payload.get("name", "")),
            columns=[str(column) for column in payload.get("columns", [])],
            data=frame,
            placeholder=bool(payload.get("placeholder", False)),
        )

    def signature(self) -> str:
        """Return a stable hash representing the table contents."""

        payload = [self.name, f"placeholder={int(self.placeholder)}"]
        if not self.data.empty:
            normalised = self.data[self.columns].copy()
            normalised = normalised.replace({pd.NA: None})
            normalised = normalised.fillna("")

            def _stringify(value: object) -> str:
                if isinstance(value, pd.Timestamp):
                    return value.isoformat()
                if isinstance(value, pd.Period):
                    return value.to_timestamp().isoformat()
                if isinstance(value, float) and pd.isna(value):
                    return "NaN"
                return str(value)

            frame_mapper = getattr(normalised, "map", None)
            if frame_mapper is not None:
                normalised = frame_mapper(_stringify)
            else:
                normalised = normalised.applymap(_stringify)
            payload.append(normalised.to_csv(index=False))
        digest = hashlib.sha1("|".join(payload).encode("utf-8")).hexdigest()
        return digest


@dataclass
class ProjectionHorizon:
    start_year: int
    end_year: int
    planning_start: str | None = None

    def __post_init__(self) -> None:
        if not self.planning_start:
            self.planning_start = f"{self.start_year:04d}-01"
        self.clamp_planning_start()

    def clamp_planning_start(self) -> None:
        """Ensure the planning start month stays within the projection horizon."""

        try:
            plan_period = pd.Period(self.planning_start, freq="M")
        except Exception:  # pragma: no cover - defensive guard
            plan_period = pd.Period(f"{self.start_year:04d}-01", freq="M")

        start_period = pd.Period(f"{self.start_year:04d}-01", freq="M")
        end_period = pd.Period(f"{self.end_year:04d}-12", freq="M")

        if plan_period < start_period:
            plan_period = start_period
        if plan_period > end_period:
            plan_period = end_period

        self.planning_start = plan_period.strftime("%Y-%m")

    @property
    def planning_start_period(self) -> pd.Period:
        return pd.Period(self.planning_start, freq="M")

    @property
    def planning_start_timestamp(self) -> pd.Timestamp:
        return self.planning_start_period.to_timestamp()

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Start Year": [self.start_year],
                "End Year": [self.end_year],
                "Planning Start": [self.planning_start],
                "Years": [self.end_year - self.start_year + 1],
            }
        )

    def signature(self) -> str:
        payload = f"{self.start_year}|{self.end_year}|{self.planning_start}"
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProjectionHorizon":
        return cls(
            start_year=int(payload.get("start_year", payload.get("Start Year", 2024))),
            end_year=int(payload.get("end_year", payload.get("End Year", 2034))),
            planning_start=payload.get("planning_start", payload.get("Planning Start")),
        )


@dataclass
class InputLandingPage:
    projection: ProjectionHorizon
    global_inputs: EditableTable
    initial_investment: EditableTable
    revenue_inputs: EditableTable
    annual_cycle_plan: EditableTable
    farm_cost_assumptions: EditableTable
    farm_capex: EditableTable
    procurement_plan: EditableTable
    product_routing: EditableTable
    commercialization_plan: EditableTable
    production_annual: EditableTable
    production_monthly: EditableTable
    direct_costs_monthly: EditableTable
    staff_positions: EditableTable
    staff_costs_monthly: EditableTable
    other_opex_monthly: EditableTable
    accounts_receivable: EditableTable
    inventory_payable: EditableTable
    loan_schedule: EditableTable
    tax_schedule: EditableTable
    inflation_schedule: EditableTable
    risk_schedule: EditableTable

    def tables(self) -> Dict[str, EditableTable]:
        return {
            "Global Inputs": self.global_inputs,
            "Initial Investment": self.initial_investment,
            "Revenue Inputs": self.revenue_inputs,
            "Annual Cycle Plan": self.annual_cycle_plan,
            "Farm Cost Assumptions": self.farm_cost_assumptions,
            "Farm Capex": self.farm_capex,
            "Procurement Plan": self.procurement_plan,
            "Product Routing": self.product_routing,
            "Commercialization Plan": self.commercialization_plan,
            "Production Annual": self.production_annual,
            "Production Monthly": self.production_monthly,
            "Direct Costs Monthly": self.direct_costs_monthly,
            "Staff Positions": self.staff_positions,
            "Staff Monthly": self.staff_costs_monthly,
            "Other Opex Monthly": self.other_opex_monthly,
            "Accounts Receivable": self.accounts_receivable,
            "Accounts Payable": self.inventory_payable,
            "Loan Schedule": self.loan_schedule,
            "Tax Schedule": self.tax_schedule,
            "Inflation Schedule": self.inflation_schedule,
            "Risk Schedule": self.risk_schedule,
        }

    def grouped_tables(self) -> "OrderedDict[str, List[EditableTable]]":
        """Return the landing-page tables grouped under the high-level sections.

        The UI and Excel exporter both rely on this method to present the
        requested categories: Global, Capex, Production, Costs, Working
        Capital, Financial, and Other Assumptions.
        """

        return OrderedDict(
            [
                ("Global", [self.global_inputs]),
                ("Capex", [self.initial_investment, self.farm_capex]),
                (
                    "Cycle Planning",
                    [
                        self.annual_cycle_plan,
                        self.production_annual,
                    ],
                ),
                ("Farming", [self.farm_cost_assumptions]),
                ("Sourcing", [self.procurement_plan]),
                ("Processing & Product Routing", [self.product_routing]),
                ("Commercialization", [self.commercialization_plan]),
                ("Legacy Compatibility", [self.production_monthly]),
                (
                    "Costs",
                    [
                        self.direct_costs_monthly,
                        self.staff_positions,
                        self.staff_costs_monthly,
                        self.other_opex_monthly,
                    ],
                ),
                (
                    "Working Capital",
                    [
                        self.accounts_receivable,
                        self.inventory_payable,
                    ],
                ),
                (
                    "Financial",
                    [
                        self.revenue_inputs,
                        self.loan_schedule,
                        self.tax_schedule,
                    ],
                ),
                (
                    "Other Assumptions",
                    [
                        self.inflation_schedule,
                        self.risk_schedule,
                    ],
                ),
            ]
        )

    def add_row(self, table_name: str, values: Dict[str, object]) -> None:
        """Add a row to one of the landing-page tables by name."""

        tables = self.tables()
        if table_name not in tables:
            raise KeyError(f"Table '{table_name}' not found. Available: {list(tables)}")
        tables[table_name].add_row(values)

    def remove_row(self, table_name: str, index: int) -> None:
        """Remove the row at *index* from a named table."""

        tables = self.tables()
        if table_name not in tables:
            raise KeyError(f"Table '{table_name}' not found. Available: {list(tables)}")
        tables[table_name].remove_row(index)

    @property
    def total_initial_investment(self) -> float:
        """Return the aggregated initial investment cost across all items."""

        data = self.initial_investment.model_frame
        return float(data.get("Cost", pd.Series(dtype=float)).sum()) if not data.empty else 0.0

    def signature(self) -> str:
        """Return a stable signature that reflects all landing-page inputs."""

        payload = [self.projection.signature()]
        for name, table in sorted(self.tables().items()):
            payload.append(f"{name}:{table.signature()}")
        return hashlib.sha1("|".join(payload).encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "InputLandingPage":
        default = default_input_page()

        def _table(field_name: str) -> EditableTable:
            table_payload = payload.get(field_name)
            if isinstance(table_payload, EditableTable):
                return table_payload
            if isinstance(table_payload, Mapping):
                return EditableTable.from_dict(table_payload)
            return getattr(default, field_name)

        projection_payload = payload.get("projection")
        if isinstance(projection_payload, ProjectionHorizon):
            projection = projection_payload
        elif isinstance(projection_payload, Mapping):
            projection = ProjectionHorizon.from_dict(projection_payload)
        else:
            projection = default.projection

        return cls(
            projection=projection,
            global_inputs=_table("global_inputs"),
            initial_investment=_table("initial_investment"),
            revenue_inputs=_table("revenue_inputs"),
            annual_cycle_plan=_table("annual_cycle_plan"),
            farm_cost_assumptions=_table("farm_cost_assumptions"),
            farm_capex=_table("farm_capex"),
            procurement_plan=_table("procurement_plan"),
            product_routing=_table("product_routing"),
            commercialization_plan=_table("commercialization_plan"),
            production_annual=_table("production_annual"),
            production_monthly=_table("production_monthly"),
            direct_costs_monthly=_table("direct_costs_monthly"),
            staff_positions=_table("staff_positions"),
            staff_costs_monthly=_table("staff_costs_monthly"),
            other_opex_monthly=_table("other_opex_monthly"),
            accounts_receivable=_table("accounts_receivable"),
            inventory_payable=_table("inventory_payable"),
            loan_schedule=_table("loan_schedule"),
            tax_schedule=_table("tax_schedule"),
            inflation_schedule=_table("inflation_schedule"),
            risk_schedule=_table("risk_schedule"),
        )


@dataclass
class ScenarioAssumption:
    name: str
    value: float
    description: str


def default_input_page() -> InputLandingPage:
    projection = ProjectionHorizon(2024, 2034, "2025-01")

    global_inputs = EditableTable(
        "Global Inputs",
        ["Parameter", "Value", "Units"],
        pd.DataFrame(
            [
                {"Parameter": "Corporate tax rate", "Value": 0.28, "Units": "%"},
                {"Parameter": "Investor share capital", "Value": 0.45, "Units": "%"},
                {"Parameter": "Owner share capital", "Value": 0.55, "Units": "%"},
                {"Parameter": "Terminal growth", "Value": 0.02, "Units": "%"},
                {"Parameter": "Capital gains tax rate", "Value": 0.05, "Units": "%"},
                {"Parameter": "Discount rate", "Value": 0.12, "Units": "%"},
                {"Parameter": "Cassava farm cost per ton", "Value": 45.0, "Units": "USD/ton"},
                {"Parameter": "Cassava purchase cost per ton", "Value": 70.0, "Units": "USD/ton"},
                {"Parameter": "Hybrid farm share", "Value": 0.5, "Units": "%"},
                {"Parameter": "Integrated cycle model enabled", "Value": 1.0, "Units": "0/1"},
                {"Parameter": "Raw cassava sorting reject %", "Value": 0.02, "Units": "%"},
                {"Parameter": "Residue recovery to feed %", "Value": 0.90, "Units": "%"},
                {"Parameter": "Farm transfer markup %", "Value": 0.08, "Units": "%"},
                {"Parameter": "Farm transfer receivable days", "Value": 30.0, "Units": "days"},
                {"Parameter": "Farm payable days", "Value": 30.0, "Units": "days"},
                {"Parameter": "Default annual production increment %", "Value": 0.04, "Units": "%"},
                {"Parameter": "Offtake floor price (USD/L)", "Value": 0.62, "Units": "USD/L"},
                {"Parameter": "Offtake ceiling price (USD/L)", "Value": 0.9, "Units": "USD/L"},
                {"Parameter": "Take-or-pay share", "Value": 0.85, "Units": "%"},
                {"Parameter": "Contracted feedstock share", "Value": 0.6, "Units": "%"},
                {"Parameter": "Open market feedstock share", "Value": 0.4, "Units": "%"},
                {"Parameter": "Contract feedstock discount", "Value": 0.08, "Units": "%"},
                {"Parameter": "Debt sculpting enabled", "Value": 0.0, "Units": "0/1"},
                {"Parameter": "Target DSCR", "Value": 1.25, "Units": "x"},
                {"Parameter": "Refinancing enabled", "Value": 0.0, "Units": "0/1"},
                {"Parameter": "Refinancing year", "Value": 2029, "Units": "Year"},
                {"Parameter": "Refinancing interest rate", "Value": 0.075, "Units": "%"},
                {"Parameter": "Repricing fee rate", "Value": 0.01, "Units": "%"},
                {"Parameter": "Break cost rate", "Value": 0.005, "Units": "%"},
                {"Parameter": "DSRA months", "Value": 6.0, "Units": "months"},
                {"Parameter": "DSCR lock-up threshold", "Value": 1.15, "Units": "x"},
                {"Parameter": "Cash sweep trigger DSCR", "Value": 1.35, "Units": "x"},
                {"Parameter": "Cash sweep share", "Value": 0.5, "Units": "%"},
                {"Parameter": "Breach cure window months", "Value": 3.0, "Units": "months"},
            ]
        ),
        placeholder=True,
    )

    initial_investment = EditableTable(
        "Initial Investment",
        ["Item", "Cost", "Life (years)", "Depreciation Rate", "Start Month"],
        pd.DataFrame(
            [
                {"Item": "Land", "Cost": 2_000_000, "Life (years)": 40, "Depreciation Rate": 0.0, "Start Month": "2024-01"},
                {"Item": "Building", "Cost": 12_000_000, "Life (years)": 25, "Depreciation Rate": None, "Start Month": "2024-01"},
                {"Item": "Plant & Equipment", "Cost": 18_000_000, "Life (years)": 15, "Depreciation Rate": None, "Start Month": "2024-01"},
                {"Item": "Farm Development", "Cost": 3_000_000, "Life (years)": 10, "Depreciation Rate": None, "Start Month": "2024-01"},
                {"Item": "EPC & Others", "Cost": 5_000_000, "Life (years)": 8, "Depreciation Rate": None, "Start Month": "2024-01"},
            ]
        ),
        placeholder=True,
    )

    revenue_inputs = EditableTable(
        "Revenue Inputs",
        ["Product", "Base Price", "Escalation", "Units"],
        pd.DataFrame(
            [
                {"Product": "Fuel Ethanol", "Base Price": 0.70, "Escalation": 0.02, "Units": "USD/L"},
                {"Product": "Animal Feed (AnFeed)", "Base Price": 120.0, "Escalation": 0.015, "Units": "USD/ton"},
            ]
        ),
        placeholder=True,
    )

    annual_cycle_plan = EditableTable(
        "Annual Cycle Plan",
        [
            "Year",
            "Cycle ID",
            "Planting Month",
            "Cultivation Months",
            "Harvest Month",
            "Processing Start Month",
            "Processing Months",
            "Cultivated Hectares",
            "Yield t/ha",
            "Field Loss %",
            "Harvest Loss %",
            "Cassava Processing Target ton",
            "Hybrid Farm Share %",
            "Annual Increment %",
        ],
        pd.DataFrame(
            [
                {
                    "Year": 2025,
                    "Cycle ID": "C1",
                    "Planting Month": "2025-01",
                    "Cultivation Months": 9,
                    "Harvest Month": "2025-09",
                    "Processing Start Month": "2025-10",
                    "Processing Months": 3,
                    "Cultivated Hectares": 5_000.0,
                    "Yield t/ha": 25.0,
                    "Field Loss %": 0.05,
                    "Harvest Loss %": 0.03,
                    "Cassava Processing Target ton": 110_000.0,
                    "Hybrid Farm Share %": 0.50,
                    "Annual Increment %": 0.04,
                }
            ]
        ),
        placeholder=True,
    )

    farm_cost_assumptions = EditableTable(
        "Farm Cost Assumptions",
        ["Cost Item", "Stage", "Basis", "Unit Cost", "Annual Increment %", "Notes"],
        pd.DataFrame(
            [
                {
                    "Cost Item": "Land Preparation",
                    "Stage": "Land Preparation",
                    "Basis": "USD/ha",
                    "Unit Cost": 180.0,
                    "Annual Increment %": 0.03,
                    "Notes": "Clearing, ploughing, ridging and field layout",
                },
                {
                    "Cost Item": "Cassava Cuttings/Seedlings",
                    "Stage": "Planting",
                    "Basis": "USD/ha",
                    "Unit Cost": 220.0,
                    "Annual Increment %": 0.03,
                    "Notes": "Certified disease-resistant planting material",
                },
                {
                    "Cost Item": "Planting Labour",
                    "Stage": "Planting",
                    "Basis": "USD/ha",
                    "Unit Cost": 120.0,
                    "Annual Increment %": 0.04,
                    "Notes": "Seasonal labour for planting operations",
                },
                {
                    "Cost Item": "Fertiliser",
                    "Stage": "Cultivation",
                    "Basis": "USD/ha",
                    "Unit Cost": 310.0,
                    "Annual Increment %": 0.04,
                    "Notes": "NPK and soil amendment programme",
                },
                {
                    "Cost Item": "Crop Protection",
                    "Stage": "Cultivation",
                    "Basis": "USD/ha",
                    "Unit Cost": 95.0,
                    "Annual Increment %": 0.04,
                    "Notes": "Herbicide, pesticide and disease management",
                },
                {
                    "Cost Item": "Weeding/Farm Maintenance",
                    "Stage": "Cultivation",
                    "Basis": "USD/ha/month",
                    "Unit Cost": 24.0,
                    "Annual Increment %": 0.04,
                    "Notes": "Routine field maintenance throughout cultivation",
                },
                {
                    "Cost Item": "Irrigation/Water",
                    "Stage": "Cultivation",
                    "Basis": "USD/ha/month",
                    "Unit Cost": 16.0,
                    "Annual Increment %": 0.035,
                    "Notes": "Water abstraction, pumping and irrigation",
                },
                {
                    "Cost Item": "Seasonal/General Field Labour",
                    "Stage": "Cultivation",
                    "Basis": "USD/ha/month",
                    "Unit Cost": 30.0,
                    "Annual Increment %": 0.04,
                    "Notes": "General labour excluding permanent farm payroll",
                },
                {
                    "Cost Item": "Machinery/Fuel",
                    "Stage": "Cultivation",
                    "Basis": "USD/ha/month",
                    "Unit Cost": 14.0,
                    "Annual Increment %": 0.04,
                    "Notes": "Tractor, implement and fuel usage",
                },
                {
                    "Cost Item": "Harvesting Labour",
                    "Stage": "Harvesting",
                    "Basis": "USD/ton harvested",
                    "Unit Cost": 9.0,
                    "Annual Increment %": 0.04,
                    "Notes": "Lifting, sorting and loading",
                },
                {
                    "Cost Item": "Farm-to-Plant Transport",
                    "Stage": "Transport",
                    "Basis": "USD/ton",
                    "Unit Cost": 8.0,
                    "Annual Increment %": 0.04,
                    "Notes": "Just-in-time delivery during processing window",
                },
                {
                    "Cost Item": "Farm Administration",
                    "Stage": "Administration",
                    "Basis": "USD/month",
                    "Unit Cost": 20_000.0,
                    "Annual Increment %": 0.04,
                    "Notes": "Agronomy support, farm office and compliance",
                },
            ]
        ),
        placeholder=True,
    )

    farm_capex = EditableTable(
        "Farm Capex",
        ["Asset", "Cost", "Life (years)", "Residual %", "Start Month", "Annual Maintenance %"],
        pd.DataFrame(
            [
                {
                    "Asset": "Land",
                    "Cost": 2_000_000.0,
                    "Life (years)": 0,
                    "Residual %": 1.0,
                    "Start Month": "2024-01",
                    "Annual Maintenance %": 0.005,
                },
                {
                    "Asset": "Farm Development",
                    "Cost": 3_000_000.0,
                    "Life (years)": 10,
                    "Residual %": 0.10,
                    "Start Month": "2024-01",
                    "Annual Maintenance %": 0.025,
                },
                {
                    "Asset": "Farm Machinery & Irrigation",
                    "Cost": 4_000_000.0,
                    "Life (years)": 12,
                    "Residual %": 0.10,
                    "Start Month": "2024-01",
                    "Annual Maintenance %": 0.04,
                },
            ]
        ),
        placeholder=True,
    )

    procurement_plan = EditableTable(
        "Procurement Plan",
        [
            "Year",
            "Contracted Share %",
            "Contract Price USD/t",
            "Open Market Price USD/t",
            "Inbound Logistics USD/t",
            "Quality Loss %",
            "Annual Price Increment %",
            "Payable Days",
        ],
        pd.DataFrame(
            [
                {
                    "Year": 2025,
                    "Contracted Share %": 0.60,
                    "Contract Price USD/t": 66.0,
                    "Open Market Price USD/t": 78.0,
                    "Inbound Logistics USD/t": 8.0,
                    "Quality Loss %": 0.03,
                    "Annual Price Increment %": 0.04,
                    "Payable Days": 30.0,
                }
            ]
        ),
        placeholder=True,
    )

    product_routing = EditableTable(
        "Product Routing",
        [
            "Stage Order",
            "Input Stream",
            "Output Stream",
            "Output Type",
            "Feedstock Grade",
            "Maximum Feedstock Age Days",
            "Reference Dry Matter %",
            "Actual Dry Matter %",
            "Allocation %",
            "Output Yield per Input",
            "Output Unit",
            "Processing Loss %",
            "Residue Yield %",
            "Monthly Capacity",
            "Processing Cost per Output Unit",
        ],
        pd.DataFrame(
            [
                {
                    "Stage Order": 1,
                    "Input Stream": "Cassava",
                    "Output Stream": "Fuel Ethanol",
                    "Output Type": "Product",
                    "Feedstock Grade": "Fresh/General",
                    "Maximum Feedstock Age Days": 7,
                    "Reference Dry Matter %": 0.30,
                    "Actual Dry Matter %": 0.30,
                    "Allocation %": 0.40,
                    "Output Yield per Input": 200.0,
                    "Output Unit": "litres",
                    "Processing Loss %": 0.02,
                    "Residue Yield %": 0.12,
                    "Monthly Capacity": 4_000_000.0,
                    "Processing Cost per Output Unit": 0.14,
                },
                {
                    "Stage Order": 1,
                    "Input Stream": "Cassava",
                    "Output Stream": "HQCF",
                    "Output Type": "Product",
                    "Feedstock Grade": "Fresh - Immediate Processing",
                    "Maximum Feedstock Age Days": 2,
                    "Reference Dry Matter %": 0.30,
                    "Actual Dry Matter %": 0.30,
                    "Allocation %": 0.15,
                    "Output Yield per Input": 0.25,
                    "Output Unit": "ton",
                    "Processing Loss %": 0.02,
                    "Residue Yield %": 0.65,
                    "Monthly Capacity": 2_500.0,
                    "Processing Cost per Output Unit": 95.0,
                },
                {
                    "Stage Order": 1,
                    "Input Stream": "Cassava",
                    "Output Stream": "Garri",
                    "Output Type": "Product",
                    "Feedstock Grade": "Fresh - Immediate Processing",
                    "Maximum Feedstock Age Days": 2,
                    "Reference Dry Matter %": 0.30,
                    "Actual Dry Matter %": 0.30,
                    "Allocation %": 0.15,
                    "Output Yield per Input": 0.25,
                    "Output Unit": "ton",
                    "Processing Loss %": 0.02,
                    "Residue Yield %": 0.65,
                    "Monthly Capacity": 2_500.0,
                    "Processing Cost per Output Unit": 110.0,
                },
                {
                    "Stage Order": 1,
                    "Input Stream": "Cassava",
                    "Output Stream": "Starch Pool",
                    "Output Type": "Intermediate",
                    "Feedstock Grade": "Large/Older - High Dry Matter",
                    "Maximum Feedstock Age Days": 14,
                    "Reference Dry Matter %": 0.30,
                    "Actual Dry Matter %": 0.35,
                    "Allocation %": 0.30,
                    "Output Yield per Input": 0.25,
                    "Output Unit": "ton",
                    "Processing Loss %": 0.02,
                    "Residue Yield %": 0.65,
                    "Monthly Capacity": 3_500.0,
                    "Processing Cost per Output Unit": 85.0,
                },
                {
                    "Stage Order": 2,
                    "Input Stream": "Starch Pool",
                    "Output Stream": "Industrial Starch",
                    "Output Type": "Product",
                    "Feedstock Grade": "Starch Intermediate",
                    "Maximum Feedstock Age Days": 0,
                    "Reference Dry Matter %": 1.00,
                    "Actual Dry Matter %": 1.00,
                    "Allocation %": 0.40,
                    "Output Yield per Input": 0.98,
                    "Output Unit": "ton",
                    "Processing Loss %": 0.01,
                    "Residue Yield %": 0.01,
                    "Monthly Capacity": 2_000.0,
                    "Processing Cost per Output Unit": 75.0,
                },
                {
                    "Stage Order": 2,
                    "Input Stream": "Starch Pool",
                    "Output Stream": "Dextrin",
                    "Output Type": "Product",
                    "Feedstock Grade": "Starch Intermediate",
                    "Maximum Feedstock Age Days": 0,
                    "Reference Dry Matter %": 1.00,
                    "Actual Dry Matter %": 1.00,
                    "Allocation %": 0.20,
                    "Output Yield per Input": 0.92,
                    "Output Unit": "ton",
                    "Processing Loss %": 0.02,
                    "Residue Yield %": 0.02,
                    "Monthly Capacity": 1_000.0,
                    "Processing Cost per Output Unit": 180.0,
                },
                {
                    "Stage Order": 2,
                    "Input Stream": "Starch Pool",
                    "Output Stream": "Glucose Syrup Pool",
                    "Output Type": "Intermediate",
                    "Feedstock Grade": "Starch Intermediate",
                    "Maximum Feedstock Age Days": 0,
                    "Reference Dry Matter %": 1.00,
                    "Actual Dry Matter %": 1.00,
                    "Allocation %": 0.40,
                    "Output Yield per Input": 1.05,
                    "Output Unit": "ton",
                    "Processing Loss %": 0.02,
                    "Residue Yield %": 0.02,
                    "Monthly Capacity": 2_000.0,
                    "Processing Cost per Output Unit": 160.0,
                },
                {
                    "Stage Order": 3,
                    "Input Stream": "Glucose Syrup Pool",
                    "Output Stream": "Glucose Syrup",
                    "Output Type": "Product",
                    "Feedstock Grade": "Glucose Syrup Intermediate",
                    "Maximum Feedstock Age Days": 0,
                    "Reference Dry Matter %": 1.00,
                    "Actual Dry Matter %": 1.00,
                    "Allocation %": 0.60,
                    "Output Yield per Input": 0.98,
                    "Output Unit": "ton",
                    "Processing Loss %": 0.01,
                    "Residue Yield %": 0.01,
                    "Monthly Capacity": 1_500.0,
                    "Processing Cost per Output Unit": 90.0,
                },
                {
                    "Stage Order": 3,
                    "Input Stream": "Glucose Syrup Pool",
                    "Output Stream": "Sorbitol",
                    "Output Type": "Product",
                    "Feedstock Grade": "Glucose Syrup Intermediate",
                    "Maximum Feedstock Age Days": 0,
                    "Reference Dry Matter %": 1.00,
                    "Actual Dry Matter %": 1.00,
                    "Allocation %": 0.40,
                    "Output Yield per Input": 0.95,
                    "Output Unit": "ton",
                    "Processing Loss %": 0.02,
                    "Residue Yield %": 0.02,
                    "Monthly Capacity": 1_000.0,
                    "Processing Cost per Output Unit": 240.0,
                },
                {
                    "Stage Order": 4,
                    "Input Stream": "Residue Pool",
                    "Output Stream": "Animal Feed",
                    "Output Type": "Product",
                    "Feedstock Grade": "Peels, Rejects & Bagasse",
                    "Maximum Feedstock Age Days": 0,
                    "Reference Dry Matter %": 1.00,
                    "Actual Dry Matter %": 1.00,
                    "Allocation %": 0.90,
                    "Output Yield per Input": 0.85,
                    "Output Unit": "ton",
                    "Processing Loss %": 0.02,
                    "Residue Yield %": 0.00,
                    "Monthly Capacity": 12_000.0,
                    "Processing Cost per Output Unit": 45.0,
                },
            ]
        ),
        placeholder=True,
    )

    commercialization_plan = EditableTable(
        "Commercialization Plan",
        [
            "Product",
            "Unit",
            "Annual Demand",
            "Annual Demand Growth %",
            "Offtake Share %",
            "Base Price",
            "Annual Price Escalation %",
            "Packaging Cost per Unit",
            "Distribution Cost per Unit",
            "Marketing Commission %",
            "Monthly Storage Loss %",
            "Storage Cost per Unit per Month",
            "Receivable Days",
            "Maximum Storage Months",
            "Inventory Valuation %",
        ],
        pd.DataFrame(
            [
                {
                    "Product": "Fuel Ethanol",
                    "Unit": "litres",
                    "Annual Demand": 22_000_000.0,
                    "Annual Demand Growth %": 0.04,
                    "Offtake Share %": 0.85,
                    "Base Price": 0.70,
                    "Annual Price Escalation %": 0.02,
                    "Packaging Cost per Unit": 0.01,
                    "Distribution Cost per Unit": 0.025,
                    "Marketing Commission %": 0.01,
                    "Monthly Storage Loss %": 0.002,
                    "Storage Cost per Unit per Month": 0.002,
                    "Receivable Days": 45.0,
                    "Maximum Storage Months": 6,
                    "Inventory Valuation %": 0.60,
                },
                {
                    "Product": "HQCF",
                    "Unit": "ton",
                    "Annual Demand": 6_000.0,
                    "Annual Demand Growth %": 0.04,
                    "Offtake Share %": 0.65,
                    "Base Price": 420.0,
                    "Annual Price Escalation %": 0.02,
                    "Packaging Cost per Unit": 22.0,
                    "Distribution Cost per Unit": 18.0,
                    "Marketing Commission %": 0.02,
                    "Monthly Storage Loss %": 0.01,
                    "Storage Cost per Unit per Month": 3.0,
                    "Receivable Days": 30.0,
                    "Maximum Storage Months": 4,
                    "Inventory Valuation %": 0.60,
                },
                {
                    "Product": "Garri",
                    "Unit": "ton",
                    "Annual Demand": 6_000.0,
                    "Annual Demand Growth %": 0.04,
                    "Offtake Share %": 0.55,
                    "Base Price": 500.0,
                    "Annual Price Escalation %": 0.02,
                    "Packaging Cost per Unit": 28.0,
                    "Distribution Cost per Unit": 22.0,
                    "Marketing Commission %": 0.025,
                    "Monthly Storage Loss %": 0.012,
                    "Storage Cost per Unit per Month": 3.5,
                    "Receivable Days": 30.0,
                    "Maximum Storage Months": 4,
                    "Inventory Valuation %": 0.60,
                },
                {
                    "Product": "Industrial Starch",
                    "Unit": "ton",
                    "Annual Demand": 4_000.0,
                    "Annual Demand Growth %": 0.04,
                    "Offtake Share %": 0.75,
                    "Base Price": 650.0,
                    "Annual Price Escalation %": 0.02,
                    "Packaging Cost per Unit": 25.0,
                    "Distribution Cost per Unit": 20.0,
                    "Marketing Commission %": 0.02,
                    "Monthly Storage Loss %": 0.004,
                    "Storage Cost per Unit per Month": 4.0,
                    "Receivable Days": 45.0,
                    "Maximum Storage Months": 8,
                    "Inventory Valuation %": 0.65,
                },
                {
                    "Product": "Dextrin",
                    "Unit": "ton",
                    "Annual Demand": 2_000.0,
                    "Annual Demand Growth %": 0.05,
                    "Offtake Share %": 0.70,
                    "Base Price": 1_050.0,
                    "Annual Price Escalation %": 0.025,
                    "Packaging Cost per Unit": 35.0,
                    "Distribution Cost per Unit": 28.0,
                    "Marketing Commission %": 0.025,
                    "Monthly Storage Loss %": 0.003,
                    "Storage Cost per Unit per Month": 5.0,
                    "Receivable Days": 45.0,
                    "Maximum Storage Months": 10,
                    "Inventory Valuation %": 0.65,
                },
                {
                    "Product": "Glucose Syrup",
                    "Unit": "ton",
                    "Annual Demand": 4_000.0,
                    "Annual Demand Growth %": 0.05,
                    "Offtake Share %": 0.75,
                    "Base Price": 850.0,
                    "Annual Price Escalation %": 0.025,
                    "Packaging Cost per Unit": 30.0,
                    "Distribution Cost per Unit": 30.0,
                    "Marketing Commission %": 0.02,
                    "Monthly Storage Loss %": 0.006,
                    "Storage Cost per Unit per Month": 6.0,
                    "Receivable Days": 45.0,
                    "Maximum Storage Months": 6,
                    "Inventory Valuation %": 0.65,
                },
                {
                    "Product": "Sorbitol",
                    "Unit": "ton",
                    "Annual Demand": 2_000.0,
                    "Annual Demand Growth %": 0.05,
                    "Offtake Share %": 0.80,
                    "Base Price": 1_250.0,
                    "Annual Price Escalation %": 0.025,
                    "Packaging Cost per Unit": 40.0,
                    "Distribution Cost per Unit": 35.0,
                    "Marketing Commission %": 0.02,
                    "Monthly Storage Loss %": 0.003,
                    "Storage Cost per Unit per Month": 7.0,
                    "Receivable Days": 60.0,
                    "Maximum Storage Months": 10,
                    "Inventory Valuation %": 0.65,
                },
                {
                    "Product": "Animal Feed",
                    "Unit": "ton",
                    "Annual Demand": 18_000.0,
                    "Annual Demand Growth %": 0.04,
                    "Offtake Share %": 0.60,
                    "Base Price": 120.0,
                    "Annual Price Escalation %": 0.015,
                    "Packaging Cost per Unit": 10.0,
                    "Distribution Cost per Unit": 15.0,
                    "Marketing Commission %": 0.02,
                    "Monthly Storage Loss %": 0.008,
                    "Storage Cost per Unit per Month": 2.0,
                    "Receivable Days": 30.0,
                    "Maximum Storage Months": 6,
                    "Inventory Valuation %": 0.50,
                },
            ]
        ),
        placeholder=True,
    )

    production_annual = EditableTable(
        "Production Annual",
        [
            "Year",
            "Start Month",
            "Cassava ton",
            "Farm Cassava ton",
            "Purchased Cassava ton",
            "Ethanol litres",
            "HQCF ton",
            "Garri ton",
            "Industrial Starch ton",
            "Dextrin ton",
            "Glucose Syrup ton",
            "Sorbitol ton",
            "Animal Feed ton",
        ],
        pd.DataFrame(
            [
                {
                    "Year": 2025,
                    "Start Month": "2025-01",
                    "Cassava ton": 110_000,
                    "Farm Cassava ton": 55_000,
                    "Purchased Cassava ton": 55_000,
                    "Ethanol litres": 22_000_000,
                    "HQCF ton": 0.0,
                    "Garri ton": 0.0,
                    "Industrial Starch ton": 0.0,
                    "Dextrin ton": 0.0,
                    "Glucose Syrup ton": 0.0,
                    "Sorbitol ton": 0.0,
                    "Animal Feed ton": 30_250,
                },
                {
                    "Year": 2026,
                    "Start Month": "2026-01",
                    "Cassava ton": 115_000,
                    "Farm Cassava ton": 57_500,
                    "Purchased Cassava ton": 57_500,
                    "Ethanol litres": 23_000_000,
                    "HQCF ton": 0.0,
                    "Garri ton": 0.0,
                    "Industrial Starch ton": 0.0,
                    "Dextrin ton": 0.0,
                    "Glucose Syrup ton": 0.0,
                    "Sorbitol ton": 0.0,
                    "Animal Feed ton": 31_625,
                },
            ]
        ),
        placeholder=True,
    )

    # Monthly production will be spread evenly by default
    monthly_index = pd.period_range("2025-01", "2025-12", freq="M")
    production_monthly = EditableTable(
        "Production Monthly",
        ["Start Month", "Cassava ton", "Ethanol litres", "Animal Feed ton", "Growth %"],
        pd.DataFrame(
            {
                "Start Month": monthly_index.astype(str),
                "Cassava ton": [10_000.0] * len(monthly_index),
                "Ethanol litres": [2_000_000.0] * len(monthly_index),
                "Animal Feed ton": [2_750.0] * len(monthly_index),
                "Growth %": [0.0] * len(monthly_index),
            }
        ),
        placeholder=True,
    )

    direct_costs_monthly = EditableTable(
        "Direct Costs Monthly",
        ["Month", "Cost Category", "Amount"],
        pd.DataFrame(
            [
                {"Month": "2025-01", "Cost Category": "Cassava Feedstock", "Amount": 600_000},
                {"Month": "2025-01", "Cost Category": "Enzymes & Chemicals", "Amount": 150_000},
                {"Month": "2025-01", "Cost Category": "Energy Cost", "Amount": 180_000},
            ]
        ),
        placeholder=True,
    )

    staff_positions = EditableTable(
        "Staff Positions",
        ["Position", "Department", "Headcount", "Monthly Salary"],
        pd.DataFrame(
            [
                {"Position": "Plant Manager", "Department": "Operations", "Headcount": 1, "Monthly Salary": 6000},
                {"Position": "Shift Supervisors", "Department": "Operations", "Headcount": 4, "Monthly Salary": 3500},
                {"Position": "Operators", "Department": "Operations", "Headcount": 40, "Monthly Salary": 1875},
                {"Position": "Field Officers", "Department": "Farming", "Headcount": 20, "Monthly Salary": 900},
                {"Position": "Farm Labour", "Department": "Farming", "Headcount": 100, "Monthly Salary": 420},
            ]
        ),
        placeholder=True,
    )

    staff_costs_monthly = EditableTable(
        "Staff Costs Monthly",
        ["Month", "Department", "Headcount", "Cost", "Annual Increment %"],
        pd.DataFrame(
            [
                {
                    "Month": "2025-01",
                    "Department": "Operations",
                    "Headcount": 45,
                    "Cost": 120_000,
                    "Annual Increment %": 0.0,
                },
                {
                    "Month": "2025-01",
                    "Department": "Farming",
                    "Headcount": 120,
                    "Cost": 65_000,
                    "Annual Increment %": 0.0,
                },
            ]
        ),
        placeholder=True,
    )

    other_opex_monthly = EditableTable(
        "Other Opex Monthly",
        ["Month", "Category", "Amount", "Annual Increment %"],
        pd.DataFrame(
            [
                {"Month": "2025-01", "Category": "Insurance", "Amount": 42_000, "Annual Increment %": 0.0},
                {"Month": "2025-01", "Category": "Service Contracts", "Amount": 30_000, "Annual Increment %": 0.0},
                {
                    "Month": "2025-01",
                    "Category": "General Administration",
                    "Amount": 82_000,
                    "Annual Increment %": 0.0,
                },
                {"Month": "2025-01", "Category": "Sales & Marketing", "Amount": 25_000, "Annual Increment %": 0.0},
                {
                    "Month": "2025-01",
                    "Category": "Research & Development",
                    "Amount": 15_000,
                    "Annual Increment %": 0.0,
                },
                {"Month": "2025-01", "Category": "Energy Cost", "Amount": 165_000, "Annual Increment %": 0.0},
            ]
        ),
        placeholder=True,
    )

    accounts_receivable = EditableTable(
        "Accounts Receivable & Other Assets",
        ["Effective Month", "Metric", "Value", "Units"],
        pd.DataFrame(
            [
                {"Effective Month": "2025-01", "Metric": "Receivables days", "Value": 45, "Units": "days"},
                {"Effective Month": "2025-01", "Metric": "Inventory days", "Value": 35, "Units": "days"},
                {"Effective Month": "2025-01", "Metric": "Prepaid expense days", "Value": 15, "Units": "days"},
                {
                    "Effective Month": "2025-01",
                    "Metric": "Other assets percent of revenue",
                    "Value": 0.02,
                    "Units": "%",
                },
            ]
        ),
        placeholder=True,
    )

    inventory_payable = EditableTable(
        "Accounts Payable",
        ["Effective Month", "Metric", "Value", "Units"],
        pd.DataFrame(
            [
                {"Effective Month": "2025-01", "Metric": "Payables days", "Value": 40, "Units": "days"},
                {"Effective Month": "2025-01", "Metric": "Other payable days", "Value": 20, "Units": "days"},
            ]
        ),
        placeholder=True,
    )

    loan_schedule = EditableTable(
        "Loan Schedule",
        [
            "Loan",
            "Type",
            "Loan Amount",
            "Base Interest",
            "Interest Rate",
            "Tenor Years",
            "Grace Years",
            "Amortization",
            "Start Month",
        ],
        pd.DataFrame(
            [
                {
                    "Loan": "Senior Debt",
                    "Type": "Term Loan",
                    "Loan Amount": 24_000_000,
                    "Base Interest": "SOFR",
                    "Interest Rate": 0.075,
                    "Tenor Years": 8,
                    "Grace Years": 1,
                    "Amortization": "Annuity",
                    "Start Month": "2024-01",
                }
            ]
        ),
        placeholder=True,
    )

    tax_schedule = EditableTable(
        "Tax Schedule",
        ["Item", "Base Rate", "Timing", "Notes"],
        pd.DataFrame(
            [
                {"Item": "Corporate income tax", "Base Rate": 0.28, "Timing": "Quarterly", "Notes": "Paid one month after quarter end"},
                {"Item": "VAT", "Base Rate": 0.07, "Timing": "Monthly", "Notes": "Input credit offset within 60 days"},
            ]
        ),
        placeholder=True,
    )

    inflation_schedule = EditableTable(
        "Inflation Schedule",
        ["Year", "CPI", "FX Index", "Tariff Escalation"],
        pd.DataFrame(
            [
                {"Year": 2024, "CPI": 0.035, "FX Index": 1.0, "Tariff Escalation": 0.0},
                {"Year": 2025, "CPI": 0.032, "FX Index": 1.02, "Tariff Escalation": 0.01},
                {"Year": 2026, "CPI": 0.03, "FX Index": 1.05, "Tariff Escalation": 0.015},
            ]
        ),
        placeholder=True,
    )

    risk_schedule = EditableTable(
        "Risk Schedule",
        [
            "Risk",
            "Class",
            "Probability",
            "Impact",
            "Expected Impact",
            "P90 Downside",
            "Duration Months",
            "Mitigation",
        ],
        pd.DataFrame(
            [
                {
                    "Risk": "Cassava yield shortfall",
                    "Class": "Volume",
                    "Probability": 0.2,
                    "Impact": "High",
                    "Expected Impact": 0.08,
                    "P90 Downside": 0.18,
                    "Duration Months": 6,
                    "Mitigation": "Crop insurance and agronomy support",
                },
                {
                    "Risk": "Ethanol price volatility",
                    "Class": "Price",
                    "Probability": 0.25,
                    "Impact": "Medium",
                    "Expected Impact": 0.06,
                    "P90 Downside": 0.14,
                    "Duration Months": 9,
                    "Mitigation": "Hedging and supply contracts",
                },
                {
                    "Risk": "Feedstock and utility inflation",
                    "Class": "Cost",
                    "Probability": 0.2,
                    "Impact": "Medium",
                    "Expected Impact": 0.05,
                    "P90 Downside": 0.12,
                    "Duration Months": 12,
                    "Mitigation": "Contracted supply and energy efficiency",
                },
                {
                    "Risk": "Construction delay",
                    "Class": "Schedule",
                    "Probability": 0.15,
                    "Impact": "High",
                    "Expected Impact": 0.1,
                    "P90 Downside": 0.22,
                    "Duration Months": 4,
                    "Mitigation": "EPC guarantees",
                },
            ]
        ),
        placeholder=True,
    )

    return InputLandingPage(
        projection=projection,
        global_inputs=global_inputs,
        initial_investment=initial_investment,
        revenue_inputs=revenue_inputs,
        annual_cycle_plan=annual_cycle_plan,
        farm_cost_assumptions=farm_cost_assumptions,
        farm_capex=farm_capex,
        procurement_plan=procurement_plan,
        product_routing=product_routing,
        commercialization_plan=commercialization_plan,
        production_annual=production_annual,
        production_monthly=production_monthly,
        direct_costs_monthly=direct_costs_monthly,
        staff_positions=staff_positions,
        staff_costs_monthly=staff_costs_monthly,
        other_opex_monthly=other_opex_monthly,
        accounts_receivable=accounts_receivable,
        inventory_payable=inventory_payable,
        loan_schedule=loan_schedule,
        tax_schedule=tax_schedule,
        inflation_schedule=inflation_schedule,
        risk_schedule=risk_schedule,
    )
