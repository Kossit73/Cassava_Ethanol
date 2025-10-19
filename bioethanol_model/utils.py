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


def irr(
    cashflows: Iterable[float],
    guess: float = 0.1,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """Internal rate of return with Newton-Raphson + bisection fallback.

    Streamlit interactions sometimes push the Newton iteration below -100% which
    previously triggered divide-by-zero errors. This version keeps the search
    inside (-1, +inf) and falls back to a robust bisection when Newton either
    diverges or encounters a flat derivative.
    """

    cashflows = list(cashflows)
    if not cashflows:
        return float("nan")

    # IRR only exists if we have at least one positive and one negative cash flow.
    has_pos = any(cf > 0 for cf in cashflows)
    has_neg = any(cf < 0 for cf in cashflows)
    if not (has_pos and has_neg):
        return float("nan")

    def _safe_power(base: float, exponent: int) -> float:
        """Return ``base**exponent`` while avoiding invalid or extreme values."""

        if exponent == 0:
            return 1.0
        if base <= 0:
            # ``base`` can dip below zero when the solver explores rates < -100%,
            # which would otherwise yield complex numbers.  Treat these cases as
            # extremely large denominators so the contribution effectively drops
            # out of the sum and the solver falls back to bisection.
            return float("inf")

        try:
            value = base**exponent
        except OverflowError:
            return np.copysign(np.finfo(float).max, base)

        if value == 0.0:
            # Clamp to the smallest normal float so divisions remain finite.
            return np.copysign(np.finfo(float).tiny, base)
        if not np.isfinite(value):
            # Extremely large values saturate at the maximum float magnitude.
            return np.copysign(np.finfo(float).max, value)
        return value

    def _npv(rate: float) -> float:
        # Keep the rate slightly above -100% to avoid division by zero.
        rate = max(rate, -0.999999)
        base = 1 + rate
        return sum(cf / _safe_power(base, i) for i, cf in enumerate(cashflows))

    def _npv_derivative(rate: float) -> float:
        rate = max(rate, -0.999999)
        base = 1 + rate
        d_val = 0.0
        for i, cf in enumerate(cashflows[1:], start=1):
            denom = _safe_power(base, i + 1)
            d_val += -i * cf / denom
        return d_val

    rate = guess
    for _ in range(max_iter):
        rate = max(rate, -0.999999)
        npv_val = _npv(rate)
        if abs(npv_val) < tol:
            return rate
        d_npv = _npv_derivative(rate)
        if d_npv == 0:
            break
        next_rate = rate - npv_val / d_npv
        if not np.isfinite(next_rate):
            break
        rate = next_rate

    # Bisection fallback – widen the upper bound until the function changes sign.
    low = -0.999999
    high = max(guess, 0.1)
    f_low = _npv(low)
    f_high = _npv(high)
    expand_iter = 0
    while f_low * f_high > 0 and expand_iter < 50:
        high *= 2.0
        f_high = _npv(high)
        expand_iter += 1
    if f_low * f_high > 0:
        return rate

    for _ in range(200):
        mid = 0.5 * (low + high)
        f_mid = _npv(mid)
        if abs(f_mid) < tol:
            return mid
        if f_low * f_mid < 0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid
    return 0.5 * (low + high)


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
