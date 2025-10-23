from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

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


@dataclass
class InputLandingPage:
    projection: ProjectionHorizon
    global_inputs: EditableTable
    initial_investment: EditableTable
    revenue_inputs: EditableTable
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
                ("Capex", [self.initial_investment]),
                (
                    "Production",
                    [
                        self.production_annual,
                        self.production_monthly,
                    ],
                ),
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
                {"Parameter": "Cassava farm cost per ton", "Value": 45.0, "Units": "USD/ton"},
                {"Parameter": "Cassava purchase cost per ton", "Value": 70.0, "Units": "USD/ton"},
                {"Parameter": "Hybrid farm share", "Value": 0.5, "Units": "%"},
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

    production_annual = EditableTable(
        "Production Annual",
        ["Year", "Start Month", "Cassava ton", "Ethanol litres", "Animal Feed ton"],
        pd.DataFrame(
            [
                {
                    "Year": 2025,
                    "Start Month": "2025-01",
                    "Cassava ton": 110_000,
                    "Ethanol litres": 22_000_000,
                    "Animal Feed ton": 30_250,
                },
                {
                    "Year": 2026,
                    "Start Month": "2026-01",
                    "Cassava ton": 115_000,
                    "Ethanol litres": 23_000_000,
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
        ["Month", "Department", "Headcount", "Cost"],
        pd.DataFrame(
            [
                {"Month": "2025-01", "Department": "Operations", "Headcount": 45, "Cost": 120_000},
                {"Month": "2025-01", "Department": "Farming", "Headcount": 120, "Cost": 65_000},
            ]
        ),
        placeholder=True,
    )

    other_opex_monthly = EditableTable(
        "Other Opex Monthly",
        ["Month", "Category", "Amount"],
        pd.DataFrame(
            [
                {"Month": "2025-01", "Category": "Insurance", "Amount": 42_000},
                {"Month": "2025-01", "Category": "Service Contracts", "Amount": 30_000},
                {"Month": "2025-01", "Category": "General Administration", "Amount": 82_000},
                {"Month": "2025-01", "Category": "Sales & Marketing", "Amount": 25_000},
                {"Month": "2025-01", "Category": "Research & Development", "Amount": 15_000},
                {"Month": "2025-01", "Category": "Energy Cost", "Amount": 165_000},
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
        ["Risk", "Probability", "Impact", "Mitigation"],
        pd.DataFrame(
            [
                {"Risk": "Cassava yield shortfall", "Probability": 0.2, "Impact": "High", "Mitigation": "Crop insurance and agronomy support"},
                {"Risk": "Ethanol price volatility", "Probability": 0.25, "Impact": "Medium", "Mitigation": "Hedging and supply contracts"},
                {"Risk": "Construction delay", "Probability": 0.15, "Impact": "High", "Mitigation": "EPC guarantees"},
            ]
        ),
        placeholder=True,
    )

    return InputLandingPage(
        projection=projection,
        global_inputs=global_inputs,
        initial_investment=initial_investment,
        revenue_inputs=revenue_inputs,
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
