import importlib.util
from pathlib import Path
import sys

import pytest

pd = pytest.importorskip("pandas")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "bioethanol_model" / "schedules.py"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_SPEC = importlib.util.spec_from_file_location("bioethanol_model.schedules", MODULE_PATH)
assert _SPEC and _SPEC.loader  # safety check for mypy/static analysers
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
compute_cost_tables = _MODULE.compute_cost_tables


def _inflation_schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Year": [2024, 2025, 2026],
            "CPI": [0.0, 0.0, 0.0],
            "FX Index": [1.0, 1.0, 1.0],
            "Tariff Escalation": [0.0, 0.0, 0.0],
        }
    )


def test_direct_costs_forward_fill_across_years():
    direct_costs = pd.DataFrame(
        [
            {"Month": "2025-01", "Cost Category": "Cassava Feedstock", "Amount": 100.0},
            {"Month": "2026-01", "Cost Category": "Cassava Feedstock", "Amount": 250.0},
            {"Month": "2025-01", "Cost Category": "Enzymes & Chemicals", "Amount": 150.0},
            {"Month": "2026-07", "Cost Category": "Enzymes & Chemicals", "Amount": 400.0},
        ]
    )

    costs = compute_cost_tables(
        direct_costs,
        pd.DataFrame(columns=["Month", "Department", "Cost"]),
        pd.DataFrame(columns=["Month", "Category", "Amount"]),
        _inflation_schedule(),
        start_year=2024,
        end_year=2026,
    )

    monthly = costs["Direct Costs"].monthly
    jan_2025 = pd.Timestamp("2025-01-01")
    dec_2025 = pd.Timestamp("2025-12-01")
    jan_2026 = pd.Timestamp("2026-01-01")
    jul_2026 = pd.Timestamp("2026-07-01")

    assert monthly.loc[jan_2025, "Cassava Feedstock"] == pytest.approx(100.0)
    assert monthly.loc[dec_2025, "Cassava Feedstock"] == pytest.approx(100.0)
    assert monthly.loc[jan_2026, "Cassava Feedstock"] == pytest.approx(250.0)

    assert monthly.loc[jan_2025, "Enzymes & Chemicals"] == pytest.approx(150.0)
    assert monthly.loc[pd.Timestamp("2026-06-01"), "Enzymes & Chemicals"] == pytest.approx(150.0)
    assert monthly.loc[jul_2026, "Enzymes & Chemicals"] == pytest.approx(400.0)

    annual = costs["Direct Costs"].annual
    assert annual.loc[2025, "Cassava Feedstock"] == pytest.approx(100.0 * 12)
    assert annual.loc[2026, "Cassava Feedstock"] == pytest.approx(250.0 * 12)
    assert annual.loc[2025, "Enzymes & Chemicals"] == pytest.approx(150.0 * 12)
    assert annual.loc[2026, "Enzymes & Chemicals"] == pytest.approx(150.0 * 6 + 400.0 * 6)


def test_staff_costs_forward_fill_and_rollup():
    staff_costs = pd.DataFrame(
        [
            {"Month": "2025-01", "Department": "Operations", "Cost": 1000.0},
            {"Month": "2026-04", "Department": "Operations", "Cost": 1200.0},
            {"Month": "2025-01", "Department": "Farming", "Cost": 500.0},
        ]
    )

    costs = compute_cost_tables(
        pd.DataFrame(columns=["Month", "Cost Category", "Amount"]),
        staff_costs,
        pd.DataFrame(columns=["Month", "Category", "Amount"]),
        _inflation_schedule(),
        start_year=2024,
        end_year=2026,
    )

    monthly = costs["Staff Costs"].monthly
    assert monthly.loc[pd.Timestamp("2025-05-01"), "Operations"] == pytest.approx(1000.0)
    assert monthly.loc[pd.Timestamp("2026-03-01"), "Operations"] == pytest.approx(1000.0)
    assert monthly.loc[pd.Timestamp("2026-04-01"), "Operations"] == pytest.approx(1200.0)
    assert monthly.loc[pd.Timestamp("2026-11-01"), "Operations"] == pytest.approx(1200.0)

    assert monthly.loc[pd.Timestamp("2025-08-01"), "Farming"] == pytest.approx(500.0)
    assert monthly.loc[pd.Timestamp("2026-12-01"), "Farming"] == pytest.approx(500.0)

    annual = costs["Staff Costs"].annual
    assert annual.loc[2025, "Operations"] == pytest.approx(1000.0 * 12)
    assert annual.loc[2026, "Operations"] == pytest.approx(1000.0 * 3 + 1200.0 * 9)
    assert annual.loc[2026, "Farming"] == pytest.approx(500.0 * 12)


def test_other_opex_forward_fill_and_rollup():
    other_opex = pd.DataFrame(
        [
            {"Month": "2025-01", "Category": "Insurance", "Amount": 42000.0},
            {"Month": "2026-02", "Category": "Insurance", "Amount": 45000.0},
            {"Month": "2025-01", "Category": "Service Contracts", "Amount": 30000.0},
            {"Month": "2026-09", "Category": "Service Contracts", "Amount": 36000.0},
        ]
    )

    costs = compute_cost_tables(
        pd.DataFrame(columns=["Month", "Cost Category", "Amount"]),
        pd.DataFrame(columns=["Month", "Department", "Cost"]),
        other_opex,
        _inflation_schedule(),
        start_year=2024,
        end_year=2026,
    )

    monthly = costs["Other Opex"].monthly
    assert monthly.loc[pd.Timestamp("2025-04-01"), "Insurance"] == pytest.approx(42000.0)
    assert monthly.loc[pd.Timestamp("2026-01-01"), "Insurance"] == pytest.approx(42000.0)
    assert monthly.loc[pd.Timestamp("2026-02-01"), "Insurance"] == pytest.approx(45000.0)
    assert monthly.loc[pd.Timestamp("2026-12-01"), "Insurance"] == pytest.approx(45000.0)

    assert monthly.loc[pd.Timestamp("2026-08-01"), "Service Contracts"] == pytest.approx(30000.0)
    assert monthly.loc[pd.Timestamp("2026-09-01"), "Service Contracts"] == pytest.approx(36000.0)

    annual = costs["Other Opex"].annual
    assert annual.loc[2025, "Insurance"] == pytest.approx(42000.0 * 12)
    assert annual.loc[2026, "Insurance"] == pytest.approx(42000.0 + 45000.0 * 11)
    assert annual.loc[2026, "Service Contracts"] == pytest.approx(30000.0 * 8 + 36000.0 * 4)
