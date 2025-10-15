from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd


def year_month_range(start_year: int, end_year: int) -> pd.DatetimeIndex:
    """Return a monthly date range inclusive of both years."""
    start = pd.Timestamp(start_year, 1, 1)
    end = pd.Timestamp(end_year, 12, 31)
    return pd.date_range(start, end, freq="MS")


def annual_periods(months: Sequence[pd.Timestamp]) -> List[pd.Timestamp]:
    """Return the first month of each year in the month index."""
    years = sorted({(m.year) for m in months})
    return [pd.Timestamp(y, 1, 1) for y in years]


def npv(rate: float, cashflows: Iterable[float]) -> float:
    cashflows = list(cashflows)
    return sum(cf / ((1 + rate) ** i) for i, cf in enumerate(cashflows))


def irr(cashflows: Iterable[float], guess: float = 0.1, tol: float = 1e-6, max_iter: int = 100) -> float:
    cashflows = list(cashflows)
    rate = guess
    for _ in range(max_iter):
        npv_val = 0.0
        d_npv = 0.0
        for i, cf in enumerate(cashflows):
            denom = (1 + rate) ** i
            npv_val += cf / denom
            if i > 0:
                d_npv += -i * cf / ((1 + rate) ** (i + 1))
        if abs(npv_val) < tol:
            return rate
        if d_npv == 0:
            break
        rate -= npv_val / d_npv
    return rate


@dataclass
class GoalSeekResult:
    target_name: str
    achieved_value: float
    tolerance: float
    iterations: int


def goal_seek(function, target: float, variable_guess: float, tol: float = 1e-6, max_iter: int = 200):
    """Simple goal seek using Newton-Raphson."""
    x = variable_guess
    step = 1e-4
    for i in range(max_iter):
        value = function(x)
        error = value - target
        if abs(error) <= tol:
            return GoalSeekResult("goal_seek", x, tol, i + 1)
        derivative = (function(x + step) - value) / step
        if derivative == 0:
            break
        x -= error / derivative
    return GoalSeekResult("goal_seek", x, tol, max_iter)
