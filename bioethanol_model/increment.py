"""Utilities for applying annual growth/decline rates to landing-page tables."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd


def _coerce_period(value: object, frequency: str) -> pd.Period | None:
    """Return a :class:`~pandas.Period` matching *frequency* for *value*.

    The helper is intentionally tolerant—strings, integers, ``Timestamp``
    objects, and ``Period`` instances are all accepted.  Invalid or missing
    values return ``None`` so
    callers can skip rows that are not tied to a calendar entry.
    """

    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None

    if isinstance(value, pd.Period):
        try:
            return value.asfreq(frequency)
        except Exception:  # pragma: no cover - defensive guard
            return None

    if isinstance(value, pd.Timestamp):
        try:
            return value.to_period(frequency)
        except Exception:  # pragma: no cover - defensive guard
            return None

    if isinstance(value, (int, np.integer)):
        try:
            return pd.Period(int(value), freq=frequency)
        except Exception:  # pragma: no cover - defensive guard
            return None

    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        try:
            return pd.Period(int(round(value)), freq=frequency)
        except Exception:  # pragma: no cover - defensive guard
            return None

    text = str(value).strip()
    if not text:
        return None

    try:
        return pd.Period(text, freq=frequency)
    except Exception:  # pragma: no cover - defensive guard
        return None


def _match_mask(series: pd.Series, value: object) -> pd.Series:
    """Return a boolean mask where *series* matches *value* (NA-safe)."""

    if series.empty:
        return pd.Series(dtype=bool, index=series.index)

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return series.isna()

    return series.fillna("<NA>").astype(str) == str(value)


def _base_value(row: pd.Series, column: str) -> float | None:
    """Return ``row[column]`` as ``float`` when possible."""

    if column not in row.index:
        return None

    value = row[column]
    if value is None:
        return None

    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if numeric is None or np.isnan(numeric):
        return None
    return float(numeric)


def apply_yearly_increment(
    df: pd.DataFrame,
    base_index: int,
    *,
    date_column: str,
    frequency: str,
    value_columns: Sequence[str],
    increments: Mapping[str, float],
    match_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return *df* with annual percentage adjustments applied.

    Parameters
    ----------
    df:
        Source dataframe containing the landing-page schedule.
    base_index:
        Row index that acts as the anchor for the increment calculation.
    date_column:
        Column providing the effective month/year for each row.
    frequency:
        Pandas frequency alias (``"M"`` or ``"Y"``) that matches *date_column*.
    value_columns:
        Sequence of numeric columns that should receive the increment.
    increments:
        Mapping of column name -> annual rate expressed as a decimal (``0.05``
        for +5%). Columns missing from this mapping are left untouched.
    match_columns:
        Optional sequence of columns whose values must match the anchor row for
        the increment to apply (e.g. category or department identifiers).
    """

    if df.empty or base_index not in df.index:
        return df.copy()

    if date_column not in df.columns:
        return df.copy()

    anchor_row = df.loc[base_index]
    anchor_period = _coerce_period(anchor_row[date_column], frequency)
    if anchor_period is None:
        return df.copy()

    relevant_columns = [col for col in value_columns if col in df.columns]
    if not relevant_columns:
        return df.copy()

    filtered_increments = {col: rate for col, rate in increments.items() if col in relevant_columns}
    if not filtered_increments:
        return df.copy()

    mask = pd.Series(True, index=df.index)
    for column in match_columns or []:
        if column in df.columns:
            mask &= _match_mask(df[column], anchor_row.get(column))

    result = df.copy()
    for idx in result.index[mask]:
        target_period = _coerce_period(result.at[idx, date_column], frequency)
        if target_period is None:
            continue

        year_delta = target_period.year - anchor_period.year
        if year_delta < 0:
            continue

        for column, rate in filtered_increments.items():
            base_val = _base_value(anchor_row, column)
            if base_val is None:
                continue

            try:
                current_val = _base_value(result.loc[idx], column)
            except KeyError:  # pragma: no cover - defensive guard
                continue

            new_value = float(base_val) * ((1.0 + float(rate)) ** year_delta)

            if current_val is None or not np.isclose(current_val, new_value, rtol=1e-9, atol=1e-9):
                result.at[idx, column] = new_value

    return result

