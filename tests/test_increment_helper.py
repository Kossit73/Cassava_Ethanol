from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "bioethanol_model" / "increment.py"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_SPEC = importlib.util.spec_from_file_location("bioethanol_model.increment", MODULE_PATH)
assert _SPEC and _SPEC.loader
increment_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(increment_module)

apply_yearly_increment = increment_module.apply_yearly_increment


def test_apply_yearly_increment_monthly_series():
    df = pd.DataFrame(
        [
            {"Month": "2025-01", "Cost Category": "Cassava Feedstock", "Amount": 100.0},
            {"Month": "2025-06", "Cost Category": "Cassava Feedstock", "Amount": 100.0},
            {"Month": "2026-01", "Cost Category": "Cassava Feedstock", "Amount": 150.0},
            {"Month": "2026-01", "Cost Category": "Enzymes & Chemicals", "Amount": 250.0},
        ]
    )

    updated = apply_yearly_increment(
        df,
        0,
        date_column="Month",
        frequency="M",
        value_columns=["Amount"],
        increments={"Amount": 0.1},
        match_columns=["Cost Category"],
    )

    assert updated.loc[0, "Amount"] == pytest.approx(100.0)
    assert updated.loc[1, "Amount"] == pytest.approx(100.0)
    assert updated.loc[2, "Amount"] == pytest.approx(110.0)

    # Ensure the other category is untouched
    assert updated.loc[3, "Amount"] == pytest.approx(250.0)


def test_apply_yearly_increment_year_frequency_multiple_columns():
    df = pd.DataFrame(
        [
            {"Year": 2024, "CPI": 0.03, "FX Index": 1.0},
            {"Year": 2025, "CPI": 0.03, "FX Index": 1.0},
            {"Year": 2026, "CPI": 0.03, "FX Index": 1.0},
        ]
    )

    updated = apply_yearly_increment(
        df,
        0,
        date_column="Year",
        frequency="Y",
        value_columns=["CPI", "FX Index"],
        increments={"CPI": -0.1, "FX Index": 0.05},
    )

    # CPI declines 10% each year relative to the base
    assert updated.loc[1, "CPI"] == pytest.approx(0.03 * (0.9))
    assert updated.loc[2, "CPI"] == pytest.approx(0.03 * (0.9**2))

    # FX Index grows 5% annually
    assert updated.loc[0, "FX Index"] == pytest.approx(1.0)
    assert updated.loc[1, "FX Index"] == pytest.approx(1.0 * 1.05)
    assert updated.loc[2, "FX Index"] == pytest.approx(1.0 * (1.05**2))


def test_apply_yearly_increment_handles_missing_columns():
    df = pd.DataFrame(
        [
            {"Month": "2025-01", "Department": "Operations", "Headcount": 10, "Cost": 1000.0},
            {"Month": "2026-01", "Department": "Operations", "Headcount": 10, "Cost": 1000.0},
        ]
    )

    updated = apply_yearly_increment(
        df,
        0,
        date_column="Month",
        frequency="M",
        value_columns=["Headcount", "Cost"],
        increments={"Headcount": 0.0, "Cost": 0.2},
        match_columns=["Department"],
    )

    assert updated.loc[1, "Headcount"] == pytest.approx(10)
    assert updated.loc[1, "Cost"] == pytest.approx(1200.0)


def test_apply_yearly_increment_extends_rows_to_horizon():
    df = pd.DataFrame(
        [
            {"Month": "2025-01", "Cost Category": "Cassava Feedstock", "Amount": 100.0},
        ]
    )

    updated = apply_yearly_increment(
        df,
        0,
        date_column="Month",
        frequency="M",
        value_columns=["Amount"],
        increments={"Amount": 0.1},
        match_columns=["Cost Category"],
        horizon_end="2027-12",
    )

    cassava_rows = (
        updated.loc[updated["Cost Category"] == "Cassava Feedstock"]
        .reset_index(drop=True)
    )

    assert list(cassava_rows["Month"]) == ["2025-01", "2026-01", "2027-01"]
    assert cassava_rows.loc[0, "Amount"] == pytest.approx(100.0)
    assert cassava_rows.loc[1, "Amount"] == pytest.approx(110.0)
    assert cassava_rows.loc[2, "Amount"] == pytest.approx(121.0)
