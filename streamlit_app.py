"""Streamlit dashboard for the Cassava bioethanol financial model."""

from __future__ import annotations

import copy
import re
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from bioethanol_model import CassavaBioethanolModel
from bioethanol_model.exporter import export_to_excel
from bioethanol_model.inputs import (
    EditableTable,
    InputLandingPage,
    ProjectionHorizon,
    default_input_page,
)
from bioethanol_model.scenario import ScenarioConfig, goal_seek_to_target, scenario_comparison
from bioethanol_model.schedules import (
    ANIMAL_FEED_TON_PER_TON,
    ETHANOL_LITRES_PER_TON,
    ExpenseSummary,
    compute_production_tables,
    compute_staff_schedule,
)
from bioethanol_model.sensitivity import (
    SensitivityScenario,
    monte_carlo_simulation,
    run_sensitivity,
    tornado_chart_inputs,
)


st.set_page_config(page_title="Cassava Bioethanol Model", layout="wide")

MODEL_VERSION_KEY = "model_version"
MC_CACHE_KEY = "mc_cache_store"
SENSITIVITY_CACHE_KEY = "sensitivity_cache"
SCENARIO_CACHE_KEY = "scenario_cache"

# Columns that are derived from other inputs and should not be editable via the
# "Modify Default Inputs & Figures" pane or the general data editor. The map is
# keyed by landing-page table name to keep the behaviour scoped and explicit.
DERIVED_COLUMN_MAP = {
    "Production Monthly": {"Ethanol litres", "Animal Feed ton"},
    "Production Annual": {"Ethanol litres", "Animal Feed ton"},
}


def _is_cassava_feedstock(value: object) -> bool:
    """Return True when *value* refers to the cassava feedstock cost row."""

    if value is None:
        return False
    text = str(value).strip().lower()
    return "cassava" in text and "feedstock" in text


# Predefined category options surfaced in the "Modify Default Inputs & Figures"
# editor. Users can still supply custom values by selecting the explicit custom
# option exposed by the editor for each table.
DIRECT_COST_CATEGORY_OPTIONS = [
    "Cassava Feedstock",
    "Enzymes & Chemicals",
    "Energy Cost",
]

CATEGORY_SELECT_OPTIONS = {
    ("Direct Costs Monthly", "Cost Category"): DIRECT_COST_CATEGORY_OPTIONS,
    ("Other Opex Monthly", "Category"): [
        "Service Contracts",
        "General Administration",
        "Research & Development",
        "Energy Cost",
        "Sales & Marketing",
    ],
    (
        "Accounts Receivable & Other Assets",
        "Metric",
    ): [
        "Receivables days",
        "Inventory days",
        "Prepaid expense days",
        "Other assets percent of revenue",
    ],
    (
        "Accounts Payable",
        "Metric",
    ): [
        "Payables days",
        "Other payable days",
    ],
}

CHANGE_BUTTON_CONFIG = {
    "Production Monthly": {
        "type": "month",
        "start_year_offset": 1,
        "help": (
            "Select the month where the new production plan should begin, click "
            "**Add change**, and then edit the newly inserted row. The model will "
            "cascade the cassava tonnage (and the derived ethanol and animal feed "
            "outputs) to all later months automatically."
        ),
    },
    "Direct Costs Monthly": {
        "type": "month",
        "start_year_offset": 1,
        "help": (
            "Pick the month the revised cost takes effect and use **Add change** to "
            "insert a row you can edit. The schedule keeps the change for that month "
            "and beyond."
        ),
    },
    "Staff Costs Monthly": {
        "type": "month",
        "start_year_offset": 1,
        "help": (
            "Choose the effective month for the staffing update, press **Add change**, "
            "and adjust the new row. Future months inherit the change unless another "
            "override is added."
        ),
    },
    "Other Opex Monthly": {
        "type": "month",
        "start_year_offset": 1,
        "help": (
            "Select the start month for the new opex figure, click **Add change**, and "
            "enter the updated values on the inserted row to carry them forward."
        ),
    },
    "Accounts Receivable & Other Assets": {
        "type": "month",
        "start_year_offset": 1,
        "help": (
            "Use **Add change** after picking the effective month to insert a new "
            "policy row. The working-capital calculations will apply it from that "
            "point onward."
        ),
    },
    "Accounts Payable": {
        "type": "month",
        "start_year_offset": 1,
        "help": (
            "Insert a new policy row by selecting the effective month and clicking "
            "**Add change**; the updated payable settings will roll forward "
            "automatically."
        ),
    },
    "Loan Schedule": {
        "type": "month",
        "start_year_offset": 0,
        "help": (
            "Choose the draw month for a facility and press **Add change** to add a "
            "new loan row tied to that date."
        ),
    },
    "Inflation Schedule": {
        "type": "year",
        "start_year_offset": 0,
        "help": (
            "Select the year that a new index value applies to, click **Add change**, "
            "and edit the inserted row."
        ),
    },
}

DEFAULT_SENSITIVITY_SCENARIOS: List[SensitivityScenario] = [
    SensitivityScenario("Corporate tax +1pp", "Corporate tax rate", 0.01),
    SensitivityScenario("Corporate tax -1pp", "Corporate tax rate", -0.01),
    SensitivityScenario("Discount rate +1pp", "Discount rate", 0.01),
    SensitivityScenario("Discount rate -1pp", "Discount rate", -0.01),
]

DEFAULT_SCENARIO_CONFIGS: List[ScenarioConfig] = [
    ScenarioConfig("Higher investor share", {"Investor share capital": 0.55}),
    ScenarioConfig("Lower investor share", {"Investor share capital": 0.35}),
    ScenarioConfig("Lower tax", {"Corporate tax rate": 0.24}),
]

TORNADO_DRIVERS: List[Tuple[str, float]] = [
    ("Corporate tax rate", 1.0),
    ("Investor share capital", 1.0),
    ("Owner share capital", 1.0),
    ("Discount rate", 1.0),
]

MONTE_CARLO_STD = {"Corporate tax rate": 0.01, "Investor share capital": 0.02}
MONTE_CARLO_ITERATIONS = 250
MONTE_CARLO_SEED = 42

PRODUCTION_EDIT_FLAG = "production_user_edit_flag"

def _load_session_inputs() -> InputLandingPage:
    """Return the mutable input landing page stored in session state."""
    if "input_page" not in st.session_state:
        st.session_state.input_page = default_input_page()
    return st.session_state.input_page


def _mark_inputs_dirty() -> None:
    """Flag the session so financial outputs are refreshed on the next run."""

    st.session_state.inputs_dirty = True


def _table_widget_key(table: EditableTable) -> str:
    """Return the Streamlit widget key for the data editor bound to *table*."""

    return f"table_editor_{table.name.replace(' ', '_').lower()}"


def _table_editor_state_key(table: EditableTable) -> str:
    """Return the session-state cache key used to mirror the table data."""

    return f"table_cache_{table.name.replace(' ', '_').lower()}"


def _sync_table_editors(page: InputLandingPage) -> None:
    """Keep Streamlit's editor cache aligned with the latest table data."""

    for table in page.tables().values():
        key = _table_editor_state_key(table)
        table_copy = table.data.copy()
        if key not in st.session_state:
            st.session_state[key] = table_copy
        else:
            editor_value = st.session_state[key]
            if not isinstance(editor_value, pd.DataFrame) or not editor_value.equals(table_copy):
                st.session_state[key] = table_copy


def _update_table_editor_state(table: EditableTable) -> None:
    """Force cached editor data to reflect *table*'s current values."""

    df_copy = table.data.copy()
    st.session_state[_table_editor_state_key(table)] = df_copy


def _reset_table_widget(table: EditableTable) -> None:
    """Remove any bound widget state so new data is rendered on the next run."""

    widget_key = _table_widget_key(table)
    if widget_key in st.session_state:
        del st.session_state[widget_key]


def _projection_month_options(projection: ProjectionHorizon, start_year_offset: int = 0) -> List[str]:
    start_year = projection.start_year + start_year_offset
    if start_year > projection.end_year:
        return []
    start_period = pd.Period(f"{start_year:04d}-01", freq="M")
    end_period = pd.Period(f"{projection.end_year:04d}-12", freq="M")
    if start_period > end_period:
        return []
    periods = pd.period_range(start_period, end_period, freq="M")
    return [p.strftime("%Y-%m") for p in periods]


def _projection_year_options(projection: ProjectionHorizon, start_year_offset: int = 0) -> List[int]:
    start_year = projection.start_year + start_year_offset
    if start_year > projection.end_year:
        return []
    return list(range(start_year, projection.end_year + 1))


def _apply_change_row(table: EditableTable, column: str, value: object) -> None:
    df = table.data.copy()
    if not df.empty:
        base_row = df.iloc[-1].to_dict()
    else:
        base_row = {col: None for col in table.columns}

    if column == "Year":
        base_row[column] = int(value)
    else:
        base_row[column] = str(value)

    if column in df.columns and not df.empty:
        mask = df[column].astype(str) == str(base_row[column])
    else:
        mask = pd.Series(False, index=df.index)

    if mask.any():
        idx = mask[mask].index[-1]
        for col in table.columns:
            df.at[idx, col] = base_row.get(col)
    else:
        df = pd.concat([df, pd.DataFrame([base_row])], ignore_index=True)

    if column == "Year" and "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        df = df.sort_values("Year", na_position="last").reset_index(drop=True)
    elif column in df.columns:
        try:
            order = pd.to_datetime(df[column].astype(str), errors="coerce").dt.to_period("M").dt.to_timestamp()
            df = df.assign(_order=order).sort_values("_order").drop(columns="_order").reset_index(drop=True)
        except Exception:  # pragma: no cover - defensive sorting guard
            df = df.reset_index(drop=True)

    table.set_data(df[table.columns], mark_user_input=True)
    _update_table_editor_state(table)
    _mark_inputs_dirty()


def _render_change_controls(page: InputLandingPage, table: EditableTable) -> bool:
    config = CHANGE_BUTTON_CONFIG.get(table.name)
    if not config:
        return False

    change_type = config["type"]
    offset = config.get("start_year_offset", 0)
    safe_key = table.name.replace(" ", "_").lower()

    if change_type == "month":
        month_col = _get_month_column(table.data)
        if not month_col:
            return False
        month_options = _projection_month_options(page.projection, offset)
        if not month_options:
            return False
        default_month = month_options[0]
        planning_start = page.projection.planning_start
        if planning_start and planning_start in month_options:
            default_month = planning_start
        selected_month = st.selectbox(
            "Change effective month",
            month_options,
            index=month_options.index(default_month) if default_month in month_options else 0,
            key=f"change_month_select_{safe_key}",
        )
        help_text = config.get("help")
        if help_text:
            st.caption(help_text)
        if st.button("Add change", key=f"change_month_btn_{safe_key}"):
            _apply_change_row(table, month_col, selected_month)
            return True
    elif change_type == "year":
        if "Year" not in table.columns:
            return False
        year_options = _projection_year_options(page.projection, offset)
        if not year_options:
            return False
        existing_years = pd.to_numeric(table.data.get("Year"), errors="coerce").dropna().astype(int)
        default_year = year_options[0]
        if not existing_years.empty:
            candidate = existing_years.max() + 1
            if candidate in year_options:
                default_year = candidate
        selected_year = st.selectbox(
            "Change effective year",
            year_options,
            index=year_options.index(int(default_year)) if default_year in year_options else 0,
            key=f"change_year_select_{safe_key}",
        )
        help_text = config.get("help")
        if help_text:
            st.caption(help_text)
        if st.button("Add change", key=f"change_year_btn_{safe_key}"):
            _apply_change_row(table, "Year", int(selected_year))
            return True

    return False


def _current_model_version() -> int:
    return int(st.session_state.get(MODEL_VERSION_KEY, 0))


def _bump_model_version() -> None:
    st.session_state[MODEL_VERSION_KEY] = _current_model_version() + 1


def _get_month_column(df: pd.DataFrame) -> str | None:
    """Return the preferred month column present in *df* (if any)."""

    for column in ("Effective Month", "Start Month", "Month"):
        if column in df.columns:
            return column
    return None


def _build_model_snapshot(page: InputLandingPage) -> tuple[CassavaBioethanolModel, Dict[str, object]]:
    """Create a model/result pair from a deep copy of the landing-page inputs."""

    snapshot = copy.deepcopy(page)
    model = CassavaBioethanolModel(snapshot)
    return model, model.build()


def _generate_excel_bytes(
    model: CassavaBioethanolModel, results: Dict[str, object], scenario: str
) -> bytes:
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir) / "Cassava_Bioethanol_Financial_Model.xlsx"
        export_to_excel(model, temp_path, results=results, scenario=scenario)
        return temp_path.read_bytes()


def _ensure_scenario_payload(
    scenario: str, snapshot: InputLandingPage
) -> Tuple[CassavaBioethanolModel, Dict[str, object]]:
    """Return (model, results) for a scenario, computing it lazily if needed."""

    payloads: Dict[str, Tuple[CassavaBioethanolModel, Dict[str, object]]] = (
        st.session_state.setdefault("scenario_payloads", {})
    )
    if scenario not in payloads:
        with st.spinner(f"Running {scenario.replace('_', ' ').title()} scenario..."):
            model = CassavaBioethanolModel(copy.deepcopy(snapshot))
            results = model.build(scenario)
        payloads[scenario] = (model, results)
        st.session_state.scenario_payloads = payloads
    return payloads[scenario]


def _update_projection(page: InputLandingPage) -> None:
    """Render projection horizon controls within the main layout."""

    st.subheader("Projection Horizon")
    start_col, end_col = st.columns(2)
    start = start_col.number_input(
        "Start Year",
        min_value=2000,
        max_value=2100,
        value=int(page.projection.start_year),
        step=1,
        key="projection_start_year",
    )
    end = end_col.number_input(
        "End Year",
        min_value=start,
        max_value=2125,
        value=int(page.projection.end_year),
        step=1,
        key="projection_end_year",
    )
    page.projection.start_year = int(start)
    page.projection.end_year = int(end)

    month_options = pd.period_range(f"{int(start):04d}-01", f"{int(end):04d}-12", freq="M")
    month_labels = [p.strftime("%Y-%m") for p in month_options]
    default_plan = page.projection.planning_start or month_labels[0]
    if default_plan not in month_labels:
        default_plan = month_labels[0]
    planning_selection = st.selectbox(
        "Planning Start Month",
        options=month_labels,
        index=month_labels.index(default_plan) if default_plan in month_labels else 0,
        key="projection_planning_start",
    )
    page.projection.planning_start = planning_selection
    page.projection.clamp_planning_start()


def _shift_month_column(table: EditableTable, year_delta: int) -> bool:
    """Shift a table's ``Month`` column by *year_delta* years.

    When the projection horizon changes we want the editable landing-page
    tables (production, staff costs, other opex) to reflect the new start
    year.  Shifting by the delta preserves the relative spacing and keeps any
    duplicate months (e.g. multiple departments in the same month) intact.
    Returns ``True`` when an update was applied.
    """

    month_col = _get_month_column(table.data)
    if year_delta == 0 or table.data.empty or month_col is None:
        return False

    try:
        periods = pd.PeriodIndex(table.data[month_col].astype(str), freq="M")
    except Exception:  # pragma: no cover - defensive parsing guard
        return False

    shifted = periods + year_delta * 12
    new_values = shifted.astype(str)
    current_values = table.data[month_col].astype(str).reset_index(drop=True)

    if current_values.equals(pd.Series(new_values)):
        return False

    table.data.loc[:, month_col] = new_values
    return True


def _sync_projection_from_session(page: InputLandingPage) -> None:
    """Ensure the landing-page projection matches the latest widget state."""

    start_key = "projection_start_year"
    end_key = "projection_end_year"
    planning_key = "projection_planning_start"

    previous_start = int(page.projection.start_year)
    previous_end = int(page.projection.end_year)
    previous_plan = str(page.projection.planning_start)

    start = int(st.session_state.get(start_key, previous_start))
    end = int(st.session_state.get(end_key, previous_end))
    if end < start:
        end = start

    page.projection.start_year = start
    page.projection.end_year = end
    st.session_state[start_key] = start
    st.session_state[end_key] = end

    month_options = pd.period_range(f"{start:04d}-01", f"{end:04d}-12", freq="M").strftime("%Y-%m").tolist()
    plan_value = str(st.session_state.get(planning_key, previous_plan))
    if plan_value not in month_options and month_options:
        plan_value = month_options[0]
    page.projection.planning_start = plan_value
    page.projection.clamp_planning_start()
    st.session_state[planning_key] = page.projection.planning_start

    year_delta = start - previous_start
    tables_to_shift = [
        page.production_monthly,
        page.direct_costs_monthly,
        page.staff_costs_monthly,
        page.other_opex_monthly,
        page.accounts_receivable,
        page.inventory_payable,
    ]

    any_shifted = False
    for tbl in tables_to_shift:
        if _shift_month_column(tbl, year_delta):
            any_shifted = True

    if (
        any_shifted
        or start != previous_start
        or end != previous_end
        or page.projection.planning_start != previous_plan
    ):
        _mark_inputs_dirty()


def _auto_compound_production(page: InputLandingPage) -> None:
    """Cascade monthly production volumes using the growth assumptions."""

    df = page.production_monthly.data.copy()
    month_col = _get_month_column(df)
    if df.empty or month_col is None:
        return

    month_series = pd.to_datetime(df[month_col].astype(str), errors="coerce")
    valid_mask = month_series.notna()
    if not valid_mask.any():
        return

    df = df.loc[valid_mask].reset_index(drop=True)
    month_periods = month_series[valid_mask].dt.to_period("M")
    df.loc[:, month_col] = month_periods.astype(str)
    month_index = month_periods

    sort_order = np.argsort(month_index.astype(str))
    if len(sort_order) and not np.all(sort_order == np.arange(len(sort_order))):
        month_index = month_index[sort_order]
        df = df.iloc[sort_order].reset_index(drop=True)

    if getattr(month_index, "has_duplicates", False) and month_index.has_duplicates:
        keep_mask = ~month_index.duplicated(keep="last")
        if keep_mask.ndim:
            keep_indices = np.flatnonzero(keep_mask)
            month_index = month_index[keep_indices]
            df = df.iloc[keep_indices].reset_index(drop=True)
        else:  # pragma: no cover - defensive fallback for scalar mask
            month_index = month_index.unique()
            df = df.iloc[: len(month_index)].reset_index(drop=True)
        df.loc[:, month_col] = month_index.astype(str)

    month_index_set = set(month_index)

    previous_cache = st.session_state.get("production_compound_cache")
    previous_series = None
    if isinstance(previous_cache, pd.DataFrame) and not previous_cache.empty:
        prev_month_col = _get_month_column(previous_cache)
        if prev_month_col and "Cassava ton" in previous_cache.columns:
            try:
                prev_index = pd.PeriodIndex(previous_cache[prev_month_col].astype(str), freq="M")
                previous_series = pd.Series(
                    pd.to_numeric(previous_cache["Cassava ton"], errors="coerce"),
                    index=prev_index,
                )
            except Exception:  # pragma: no cover - defensive parsing guard
                previous_series = None

    numeric_columns = [
        col for col in ("Cassava ton", "Ethanol litres", "Animal Feed ton") if col in df.columns
    ]
    manual_columns = [col for col in ("Cassava ton",) if col in df.columns]
    if not numeric_columns:
        return
    if not manual_columns:
        manual_columns = numeric_columns[:1]

    growth_col = next((c for c in df.columns if "growth" in c.lower()), None)
    growth_values = pd.Series(dtype=float)
    if growth_col:
        growth_values = pd.to_numeric(df[growth_col], errors="coerce")
        growth_values.index = month_index
        if not growth_values.dropna().empty and growth_values.nunique(dropna=False) <= 1:
            base_growth = float(growth_values.dropna().iloc[0]) if not growth_values.dropna().empty else 0.0
            growth_values = pd.Series(
                [base_growth] + [np.nan] * (len(growth_values) - 1), index=month_index
            )
    else:
        growth_values = pd.Series(index=month_index, dtype=float)

    manual_periods: set[pd.Period] = set()
    planning_period: pd.Period | None = None
    first_period: pd.Period | None = None
    manual_state_key = "production_manual_periods"
    session_manual = st.session_state.get(manual_state_key, set())
    if not isinstance(session_manual, set):
        session_manual = set(session_manual)
    manual_session_periods: set[pd.Period] = set()
    for value in session_manual:
        try:
            manual_session_periods.add(pd.Period(str(value), freq="M"))
        except Exception:  # pragma: no cover - defensive parsing guard
            continue
    if getattr(page.projection, "planning_start", None):
        try:
            planning_period = pd.Period(page.projection.planning_start, freq="M")
        except Exception:  # pragma: no cover - defensive parsing guard
            planning_period = None

    if len(month_index) > 0:
        for candidate in month_index:
            if planning_period is not None and candidate < planning_period:
                continue
            first_period = candidate
            break
        if first_period is None:
            first_period = month_index[0]

    if not growth_values.empty:
        # Treat the first row as the anchor growth rate. Any subsequent entries
        # that simply repeat this base value are considered placeholders and can
        # be overridden by the compounded series. If a user enters a different
        # growth figure for a later month we keep it so that a new cascade can
        # start from that point.
        base_period = growth_values.index[0]
        base_growth = float(growth_values.iloc[0]) if pd.notna(growth_values.iloc[0]) else 0.0
        growth_values.iloc[0] = base_growth
        tolerance = 1e-9
        for idx in growth_values.index[1:]:
            raw_val = growth_values.at[idx]
            if isinstance(raw_val, (pd.Series, np.ndarray, list, tuple)):
                if len(raw_val) == 0:
                    val = np.nan
                else:
                    val = raw_val[0]
            else:
                val = raw_val

            try:
                is_missing = pd.isna(val)
            except ValueError:
                # pandas can raise when the scalar resolution is ambiguous;
                # treat it as missing so the cascade can overwrite it.
                is_missing = True

            if is_missing:
                continue

            try:
                numeric_val = float(val)
            except (TypeError, ValueError):
                continue

            if abs(numeric_val - base_growth) < tolerance:
                growth_values.at[idx] = np.nan
        manual_periods.add(base_period)
        if planning_period is not None and base_period < planning_period:
            manual_periods.discard(base_period)

    growth_periods: set[pd.Period] = set()
    for period, val in growth_values.dropna().items():
        if val is not None:
            growth_periods.add(period)

    cassava_current = pd.Series(dtype=float)
    if "Cassava ton" in df.columns:
        cassava_current = pd.to_numeric(df["Cassava ton"], errors="coerce")
        cassava_current.index = month_index

    changed_periods: set[pd.Period] = set()
    if previous_series is None:
        if first_period is not None:
            changed_periods.add(first_period)
    else:
        aligned_prev = previous_series.reindex(month_index)
        for period, value in cassava_current.items():
            prev_val = aligned_prev.get(period)

            if isinstance(prev_val, (pd.Series, np.ndarray, list, tuple)):
                prev_series = pd.Series(prev_val).dropna()
                prev_val = prev_series.iloc[0] if not prev_series.empty else np.nan

            try:
                current_missing = pd.isna(value)
            except ValueError:
                current_missing = True

            try:
                prev_missing = pd.isna(prev_val)
            except ValueError:
                prev_missing = True

            if current_missing:
                if not prev_missing:
                    changed_periods.add(period)
                continue

            try:
                numeric_val = float(value)
            except (TypeError, ValueError):
                continue

            if prev_missing or not np.isfinite(prev_val) or not np.isclose(float(prev_val), numeric_val, atol=1e-9):
                changed_periods.add(period)

    baseline_changed = False
    if first_period is not None:
        if previous_series is None:
            baseline_changed = True
        elif first_period in changed_periods:
            baseline_changed = True

    if baseline_changed:
        preserve = {p for p in changed_periods if p != first_period}
        manual_periods = set(preserve)
    else:
        manual_periods = set(manual_session_periods)
        manual_periods.update(changed_periods)

    manual_periods.update(growth_periods)

    if planning_period is not None:
        manual_periods = {p for p in manual_periods if p >= planning_period}

    if first_period is not None:
        manual_periods.add(first_period)

    manual_periods &= month_index_set

    st.session_state[manual_state_key] = {
        period.strftime("%Y-%m") for period in manual_periods
    }

    seed_df = df.copy()
    manual_mask = month_index.isin(list(manual_periods))
    for col in manual_columns:
        seed_df.loc[~manual_mask, col] = np.nan
    if growth_col:
        seed_df[growth_col] = growth_values.values

    production = compute_production_tables(
        seed_df,
        page.projection.start_year,
        page.projection.end_year,
        planning_start=page.projection.planning_start_timestamp,
    )

    monthly = production.monthly.copy()
    if monthly.empty:
        return

    monthly_reset = monthly.reset_index()
    monthly_reset["Month"] = monthly_reset["Month"].dt.to_period("M").astype(str)
    if month_col != "Month":
        monthly_reset[month_col] = monthly_reset["Month"]
        monthly_reset = monthly_reset.drop(columns=["Month"])
    if growth_col:
        display_growth = growth_values.copy()
        if isinstance(display_growth.index, pd.PeriodIndex):
            display_growth.index = display_growth.index.to_timestamp()
        if not display_growth.index.is_unique:
            display_growth = display_growth[~display_growth.index.duplicated(keep="last")]
        display_growth = display_growth.reindex(monthly.index).ffill()
        monthly_reset[growth_col] = display_growth.values

    monthly_order = [col for col in df.columns if col in monthly_reset.columns]
    new_monthly = monthly_reset[monthly_order].copy()
    for col in numeric_columns:
        if col in new_monthly.columns:
            new_monthly[col] = pd.to_numeric(new_monthly[col], errors="coerce").round(6)
    if growth_col and growth_col in new_monthly.columns:
        new_monthly[growth_col] = pd.to_numeric(new_monthly[growth_col], errors="coerce")

    current_monthly = df[monthly_order].copy()
    for col in numeric_columns:
        if col in current_monthly.columns:
            current_monthly[col] = pd.to_numeric(current_monthly[col], errors="coerce").round(6)
    if growth_col and growth_col in current_monthly.columns:
        current_monthly[growth_col] = pd.to_numeric(current_monthly[growth_col], errors="coerce")

    updated = False
    user_driven_change = previous_series is not None and bool(changed_periods)
    user_edit_flag = bool(st.session_state.get(PRODUCTION_EDIT_FLAG, False))

    if not new_monthly.equals(current_monthly):
        mark_user = user_edit_flag or user_driven_change or not page.production_monthly.placeholder
        page.production_monthly.set_data(new_monthly, mark_user_input=mark_user)
        _update_table_editor_state(page.production_monthly)
        _reset_table_widget(page.production_monthly)
        updated = True

    st.session_state["production_compound_cache"] = new_monthly.copy()

    annual = production.annual.copy()
    annual.index.name = "Year"
    annual_reset = annual.reset_index()
    annual_order = [col for col in page.production_annual.data.columns if col in annual_reset.columns]
    new_annual = annual_reset[annual_order].copy()
    for col in annual_order:
        if col != "Year" and col in new_annual.columns:
            new_annual[col] = pd.to_numeric(new_annual[col], errors="coerce").round(6)

    current_annual = page.production_annual.data[annual_order].copy()
    for col in annual_order:
        if col != "Year" and col in current_annual.columns:
            current_annual[col] = pd.to_numeric(current_annual[col], errors="coerce").round(6)

    if not new_annual.equals(current_annual):
        mark_user = user_edit_flag or user_driven_change or not page.production_annual.placeholder
        page.production_annual.set_data(new_annual, mark_user_input=mark_user)
        _update_table_editor_state(page.production_annual)
        _reset_table_widget(page.production_annual)
        updated = True

    if updated:
        _mark_inputs_dirty()


def _apply_growth_cascade(
    page: InputLandingPage,
    table: EditableTable,
    df: pd.DataFrame,
    row_idx: int,
    growth_percent: float,
) -> tuple[pd.DataFrame, Tuple[float, float] | None]:
    """Apply *growth_percent* (entered as percentage) to cascade production."""

    if df.empty or row_idx not in df.index:
        return df, None

    month_col = _get_month_column(df)
    if month_col is None or "Cassava ton" not in df.columns:
        return df, None

    month_values = pd.to_datetime(df[month_col].astype(str), errors="coerce").dt.to_period("M")
    if month_values.isna().all():
        return df, None

    try:
        base_period = month_values.iloc[row_idx]
    except Exception:  # pragma: no cover - defensive guard
        return df, None

    if pd.isna(base_period):
        return df, None

    rate_decimal = float(growth_percent)
    if not np.isfinite(rate_decimal):
        rate_decimal = 0.0
    if abs(rate_decimal) > 1.0:
        rate_decimal = rate_decimal / 100.0
    rate_decimal = float(np.clip(rate_decimal, -0.99, 10.0))

    working = df.copy()
    cassava_col = "Cassava ton"
    working[cassava_col] = pd.to_numeric(working[cassava_col], errors="coerce")
    growth_col = next((c for c in working.columns if "growth" in c.lower()), None)

    later_mask = month_values > base_period
    working.loc[later_mask, cassava_col] = np.nan

    if growth_col:
        working[growth_col] = pd.to_numeric(working[growth_col], errors="coerce")
        working.at[row_idx, growth_col] = rate_decimal
        working.loc[later_mask, growth_col] = np.nan

    table.set_data(working, mark_user_input=True)
    _update_table_editor_state(table)
    st.session_state[PRODUCTION_EDIT_FLAG] = True
    _mark_inputs_dirty()

    _auto_compound_production(page)

    refreshed = page.production_monthly.data.copy()
    derived: Tuple[float, float] | None = None
    refreshed_month_col = _get_month_column(refreshed)
    if refreshed_month_col and "Cassava ton" in refreshed.columns:
        month_str = base_period.strftime("%Y-%m")
        match = refreshed[refreshed[refreshed_month_col] == month_str]
        if not match.empty:
            cassava_val = pd.to_numeric(match["Cassava ton"], errors="coerce").iloc[0]
            if pd.notna(cassava_val):
                ethanol_val = float(cassava_val) * ETHANOL_LITRES_PER_TON
                feed_val = float(cassava_val) * ANIMAL_FEED_TON_PER_TON
                derived = (ethanol_val, feed_val)

    return refreshed, derived


def _update_staff_costs_from_positions(page: InputLandingPage) -> None:
    """Keep the monthly staff cost table aligned with position salaries."""

    if page.staff_positions.placeholder:
        return

    staff_df = page.staff_costs_monthly.data.copy()
    if staff_df.empty or "Department" not in staff_df.columns:
        return

    schedule = compute_staff_schedule(page.staff_positions.model_frame)
    summary = schedule.department_summary
    if summary.empty or "Average Monthly Salary" not in summary.columns:
        return

    avg_salary = summary.set_index("Department")["Average Monthly Salary"].to_dict()
    staff_df["Headcount"] = pd.to_numeric(staff_df.get("Headcount"), errors="coerce").fillna(0.0)

    updated_costs = []
    for _, row in staff_df.iterrows():
        dept = row.get("Department")
        salary = avg_salary.get(dept)
        if salary is None or not np.isfinite(salary):
            try:
                current_cost = float(row.get("Cost", 0.0))
            except (TypeError, ValueError):
                current_cost = 0.0
            updated_costs.append(current_cost)
        else:
            updated_costs.append(float(row.get("Headcount", 0.0)) * float(salary))

    staff_df["Cost"] = updated_costs
    page.staff_costs_monthly.set_data(staff_df, mark_user_input=True)


def _update_feedstock_costs(page: InputLandingPage, scenario: str) -> None:
    """Recalculate cassava feedstock costs using the active scenario pricing."""

    if page.direct_costs_monthly.data.empty or page.direct_costs_monthly.placeholder:
        return

    direct_df = page.direct_costs_monthly.data.copy()
    if direct_df.empty or "Cost Category" not in direct_df.columns:
        return

    feed_mask = direct_df["Cost Category"].astype(str).str.contains("cassava", case=False, na=False)
    if not feed_mask.any():
        return

    global_df = page.global_inputs.data
    if global_df.empty or "Parameter" not in global_df.columns:
        return

    lookup = global_df.set_index("Parameter")["Value"].to_dict()

    def _get_global(name: str, default: float) -> float:
        try:
            return float(lookup.get(name, default))
        except (TypeError, ValueError):
            return default

    farm_cost = _get_global("Cassava farm cost per ton", 45.0)
    purchase_cost = _get_global("Cassava purchase cost per ton", 70.0)
    farm_share = float(np.clip(_get_global("Hybrid farm share", 0.5), 0.0, 1.0))

    scenario = (scenario or "FARM_ONLY").upper()
    if scenario == "FARM_ONLY":
        cost_per_ton = farm_cost
    elif scenario == "BUY_ONLY":
        cost_per_ton = purchase_cost
    else:
        cost_per_ton = farm_share * farm_cost + (1 - farm_share) * purchase_cost

    production_source = page.production_monthly.model_frame
    if production_source.empty:
        return

    production = compute_production_tables(
        production_source,
        page.projection.start_year,
        page.projection.end_year,
        planning_start=page.projection.planning_start_timestamp,
    )
    cassava_series = pd.to_numeric(
        production.monthly.get("Cassava ton", pd.Series(dtype=float)), errors="coerce"
    ).fillna(method="ffill").fillna(method="bfill")
    if cassava_series.empty:
        return

    cassava_series.index = cassava_series.index.to_period("M").to_timestamp()
    fallback = float(cassava_series.mean()) if not cassava_series.empty else 0.0

    def _month_to_timestamp(value: object) -> pd.Timestamp | None:
        try:
            return pd.Period(str(value), freq="M").to_timestamp()
        except Exception:  # pragma: no cover - defensive parsing guard
            return None

    direct_df = direct_df.copy()
    month_stamps = direct_df["Month"].apply(_month_to_timestamp)
    updated_amounts = []
    for is_feed, month, current in zip(feed_mask, month_stamps, direct_df["Amount"]):
        if not is_feed:
            updated_amounts.append(current)
            continue
        tons = fallback
        if month is not None and month in cassava_series.index:
            tons = float(cassava_series.loc[month])
        updated_amounts.append(tons * cost_per_ton)

    direct_df["Amount"] = updated_amounts
    page.direct_costs_monthly.set_data(direct_df, mark_user_input=True)


def _sync_working_capital_tables(page: InputLandingPage) -> None:
    """Refresh the Accounts Payable editor state after landing-page edits."""

    inv_table = page.inventory_payable
    if inv_table is None:
        return

    # With payables metrics now maintained exclusively in the Accounts Payable table,
    # we only need to refresh the editor cache so user edits remain visible. The
    # values themselves come directly from the landing-page entries.
    _update_table_editor_state(inv_table)


def _apply_dependent_updates(page: InputLandingPage, scenario: str) -> None:
    """Ensure derived landing-page tables stay synchronised with inputs."""

    _auto_compound_production(page)
    _update_staff_costs_from_positions(page)
    _update_feedstock_costs(page, scenario)
    _sync_working_capital_tables(page)


def _key_assumptions_controls(table: EditableTable) -> None:
    """Expose frequently tweaked assumptions inside the main page."""

    st.subheader("Key Assumptions")
    original_df = table.data.copy()
    df = table.data.copy()
    updated = False
    slider_cfg = {
        "Corporate tax rate": dict(min_value=0.0, max_value=0.7, step=0.01),
        "Investor share capital": dict(min_value=0.0, max_value=1.0, step=0.01),
        "Owner share capital": dict(min_value=0.0, max_value=1.0, step=0.01),
        "Discount rate": dict(min_value=0.0, max_value=0.5, step=0.01),
    }

    for parameter, cfg in slider_cfg.items():
        if parameter in df["Parameter"].values:
            idx = df.index[df["Parameter"] == parameter][0]
            state_key = f"key_assumption_{parameter.replace(' ', '_').lower()}"
            value_key = f"{state_key}_value"
            min_value = float(cfg["min_value"])
            max_value = float(cfg["max_value"])
            step = float(cfg["step"])

            current_default = (
                float(df.at[idx, "Value"]) if pd.notna(df.at[idx, "Value"]) else min_value
            )
            if value_key not in st.session_state:
                st.session_state[value_key] = current_default

            original_value = current_default

            current_value = st.number_input(
                parameter,
                min_value=min_value,
                max_value=max_value,
                step=float(step),
                value=float(st.session_state[value_key]),
                key=f"{state_key}_input",
            )
            st.session_state[value_key] = float(current_value)
            df.at[idx, "Value"] = float(current_value)
            if not np.isclose(original_value, float(current_value)):
                updated = True
                _mark_inputs_dirty()
    table.set_data(df, mark_user_input=updated)
    if not df.equals(original_df):
        _update_table_editor_state(table)


def _numeric_step(value: float) -> float:
    """Return a sensible increment for Streamlit number inputs."""

    if value is None or pd.isna(value):
        return 0.01
    value = abs(float(value))
    if value == 0:
        return 0.01
    exponent = max(-2, int(np.floor(np.log10(value))) - 1)
    return round(10 ** exponent, 6)


def _display_production_metrics(derived_metrics: Tuple[float, float] | None) -> None:
    """Render helper metrics for production table edits."""

    if derived_metrics is None:
        st.info(
            "Enter a cassava tonnage to calculate the ethanol and animal feed outputs for this row."
        )
        return

    ethanol_val, feed_val = derived_metrics
    metric_cols = st.columns(2)
    metric_cols[0].metric(
        "Calculated Ethanol (litres)",
        f"{ethanol_val:,.0f}",
    )
    metric_cols[1].metric(
        "Calculated Animal Feed (ton)",
        f"{feed_val:,.3f}",
    )


def _row_editor_form(
    table: EditableTable,
    row_idx: int,
    projection: ProjectionHorizon,
    *,
    widget_prefix: str,
) -> Tuple[pd.DataFrame, bool, Tuple[float, float] | None]:
    """Shared routine to edit a single row within a landing-page table."""

    df = table.data.copy()
    if df.empty or row_idx not in df.index:
        return df, False, None

    original_row = df.loc[row_idx].copy()

    read_only_row = (
        table.name == "Direct Costs Monthly"
        and "Cost Category" in df.columns
        and _is_cassava_feedstock(df.at[row_idx, "Cost Category"])
    )

    if read_only_row:
        st.info(
            "Cassava feedstock costs are scenario-driven. Review the values below; they are locked for editing."
        )

    month_range = pd.period_range(
        f"{int(projection.start_year):04d}-01",
        f"{int(projection.end_year):04d}-12",
        freq="M",
    )
    month_options = [p.strftime("%Y-%m") for p in month_range]

    derived_columns = DERIVED_COLUMN_MAP.get(table.name, set())
    derived_metrics: Tuple[float, float] | None = None
    updated = False

    for column in table.columns:
        current_value = df.at[row_idx, column]
        widget_key = f"{widget_prefix}_{table.name}_{row_idx}_{column}".replace(" ", "_").lower()

        if read_only_row:
            display_value = ""
            if current_value is not None and not (
                isinstance(current_value, float) and pd.isna(current_value)
            ):
                display_value = str(current_value)
            st.text_input(
                column,
                value=display_value,
                key=widget_key,
                disabled=True,
            )
            continue

        if column in derived_columns:
            if table.name == "Production Monthly":
                # Skip rendering derived production outputs so the focused editor
                # only exposes the cassava driver and growth inputs.
                continue

            display_value = ""
            if current_value is not None and not (isinstance(current_value, float) and pd.isna(current_value)):
                if isinstance(current_value, (int, float, np.floating, np.integer)):
                    display_value = f"{float(current_value):,.6f}".rstrip("0").rstrip(".")
                else:
                    display_value = str(current_value)
            st.text_input(
                column,
                value=display_value,
                key=widget_key,
                disabled=True,
                help="This value is calculated automatically from other inputs.",
            )
            continue

        column_label = column.lower().replace("_", " ")
        if re.search(r"\bmonth\b", column_label):
            current_str = (
                None
                if current_value is None or (isinstance(current_value, float) and pd.isna(current_value))
                else str(current_value)
            )

            option_list = ["Not set"] + month_options
            if current_str and current_str not in option_list:
                option_list.insert(1, current_str)

            default_index = 0
            if current_str and current_str in option_list:
                default_index = option_list.index(current_str)

            selection = st.selectbox(
                column,
                options=option_list,
                index=default_index,
                key=widget_key,
            )

            new_value = None if selection == "Not set" else selection
            df.at[row_idx, column] = new_value
            if current_str != new_value:
                updated = True
            continue

        category_key = (table.name, column)
        if category_key in CATEGORY_SELECT_OPTIONS:
            options = list(CATEGORY_SELECT_OPTIONS[category_key])
            current_str = (
                ""
                if current_value is None or (isinstance(current_value, float) and pd.isna(current_value))
                else str(current_value)
            )
            if current_str and current_str not in options:
                options.append(current_str)
            custom_label = "Custom value"
            if custom_label not in options:
                options.append(custom_label)

            default_index = 0
            if current_str and current_str in options:
                default_index = options.index(current_str)
            elif custom_label in options and current_str and current_str not in CATEGORY_SELECT_OPTIONS[category_key]:
                default_index = options.index(current_str)

            selection = st.selectbox(
                column,
                options=options,
                index=default_index,
                key=widget_key,
            )

            if selection == custom_label:
                custom_key = f"{widget_prefix}_{table.name}_{row_idx}_{column}_custom".replace(" ", "_").lower()
                custom_value = st.text_input(
                    f"Specify {column.lower()}",
                    value=current_str if current_str not in CATEGORY_SELECT_OPTIONS[category_key] else "",
                    key=custom_key,
                )
                new_value = custom_value.strip() if custom_value.strip() else None
            else:
                new_value = selection

            df.at[row_idx, column] = new_value
            if (current_str or None) != (new_value or None):
                updated = True
            continue

        numeric_series = pd.to_numeric(df[column], errors="coerce")
        is_numeric = pd.api.types.is_numeric_dtype(df[column]) or numeric_series.notna().any()

        if is_numeric:
            if row_idx in numeric_series.index:
                base_value = numeric_series.loc[row_idx]
            else:
                base_value = numeric_series.iloc[0] if not numeric_series.empty else 0.0
            if pd.isna(base_value):
                base_value = 0.0
            original_value = float(base_value)
            step = float(_numeric_step(base_value))
            number_format = "%.0f" if step >= 1 else "%.4f"
            new_value = st.number_input(
                column,
                value=float(base_value),
                step=step,
                format=number_format,
                key=widget_key,
            )
            if pd.api.types.is_integer_dtype(df[column]):
                new_value = int(round(new_value))
            df.at[row_idx, column] = new_value
            if not np.isclose(original_value, float(new_value)):
                updated = True
        else:
            text_value = "" if current_value is None or pd.isna(current_value) else str(current_value)
            new_value = st.text_input(
                column,
                value=text_value,
                key=widget_key,
            )
            df.at[row_idx, column] = new_value
            if str(new_value) != str(text_value):
                updated = True

    if table.name == "Production Monthly" and "Cassava ton" in df.columns:
        cassava_raw = df.at[row_idx, "Cassava ton"]
        cassava_val = pd.to_numeric(pd.Series([cassava_raw]), errors="coerce").iloc[0]
        if pd.notna(cassava_val):
            ethanol_val = float(cassava_val) * ETHANOL_LITRES_PER_TON
            feed_val = float(cassava_val) * ANIMAL_FEED_TON_PER_TON
            if "Ethanol litres" in df.columns:
                df.at[row_idx, "Ethanol litres"] = ethanol_val
            if "Animal Feed ton" in df.columns:
                df.at[row_idx, "Animal Feed ton"] = feed_val
            derived_metrics = (ethanol_val, feed_val)

    if not df.loc[row_idx].equals(original_row):
        updated = True

    if updated and table.name == "Production Monthly":
        st.session_state[PRODUCTION_EDIT_FLAG] = True

    return df, updated, derived_metrics


def _modify_default_inputs(page: InputLandingPage) -> None:
    """Allow users to tweak any default input/figure via focused controls."""

    st.subheader("Modify Default Inputs & Figures")
    tables = page.tables()
    if not tables:
        st.info("No tables are available to edit.")
        return

    table_names = list(tables.keys())
    table_name = st.selectbox(
        "Select table",
        table_names,
        key="default_table_select",
    )
    table = tables[table_name]
    df = table.data.copy()

    if df.empty:
        st.info("The selected table has no rows to modify. Use the table editor below to add data.")
        return

    id_column = table.columns[0] if table.columns else None
    row_indices = list(df.index)

    def _format_row(idx: int) -> str:
        if id_column and id_column in table.data.columns:
            label = table.data.at[idx, id_column]
            if label is None or pd.isna(label) or str(label).strip() == "":
                label = f"Row {idx + 1}"
        else:
            label = f"Row {idx + 1}"

        if (
            table.name == "Direct Costs Monthly"
            and "Cost Category" in table.data.columns
        ):
            category = table.data.at[idx, "Cost Category"]
            if category and not pd.isna(category):
                label = f"{label} – {category}"

        return f"{idx + 1}. {label}"

    row_idx = st.selectbox(
        "Select row",
        row_indices,
        format_func=_format_row,
        key=f"default_row_select_{table_name}",
    )

    st.markdown("Adjust the values for the selected row:")

    df_updated, updated, derived_metrics = _row_editor_form(
        table,
        row_idx,
        page.projection,
        widget_prefix="default_edit",
    )

    cascade_applied = False
    cascade_metrics: Tuple[float, float] | None = None

    if table.name == "Production Monthly":
        _display_production_metrics(derived_metrics)

        growth_col = next((c for c in df_updated.columns if "growth" in c.lower()), None)
        default_growth_pct = 0.0
        if growth_col and row_idx in df_updated.index:
            current_growth = pd.to_numeric(pd.Series([df_updated.at[row_idx, growth_col]]), errors="coerce").iloc[0]
            if pd.notna(current_growth):
                default_growth_pct = float(current_growth) * 100.0

        st.markdown("---")
        st.markdown("**Cascade growth across projection horizon**")
        st.caption(
            "Set a monthly growth percentage to apply from this month onward. The model will "
            "recalculate all subsequent production months automatically."
        )
        growth_input = st.number_input(
            "Monthly growth % to apply",
            value=float(default_growth_pct),
            step=0.1,
            format="%.2f",
            key=f"growth_pct_apply_{table.name.replace(' ', '_')}_{row_idx}",
        )
        if st.button(
            "Apply growth to remaining months",
            key=f"apply_growth_btn_{table.name.replace(' ', '_')}_{row_idx}",
        ):
            df_updated, cascade_metrics = _apply_growth_cascade(
                page,
                table,
                df_updated,
                row_idx,
                growth_input,
            )
            cascade_applied = True
            if cascade_metrics is not None:
                derived_metrics = cascade_metrics

    st.caption("Updates are applied immediately. Use the section tables below for bulk edits or row management.")
    if cascade_applied:
        df = table.data.copy()
        if not df.equals(df_updated):
            table.set_data(df, mark_user_input=True)
        _update_table_editor_state(table)
    elif updated:
        table.set_data(df_updated, mark_user_input=True)
        _mark_inputs_dirty()
        _update_table_editor_state(table)
    else:
        # Ensure the focused editor stays aligned even when only formatting changes occur.
        _update_table_editor_state(table)


def _apply_production_delta(
    page: InputLandingPage, table: EditableTable, row_idx: int, delta: float
) -> bool:
    """Adjust the cassava tonnage for *row_idx* by *delta* and refresh derived fields."""

    if table.name != "Production Monthly" or "Cassava ton" not in table.columns:
        return False

    df = table.data.copy()
    if df.empty or row_idx not in df.index:
        return False

    cassava_raw = pd.to_numeric(pd.Series([df.at[row_idx, "Cassava ton"]]), errors="coerce").iloc[0]
    current_value = float(cassava_raw) if pd.notna(cassava_raw) else 0.0
    new_value = max(0.0, current_value + float(delta))

    if not np.isfinite(new_value) or np.isclose(new_value, current_value, atol=1e-9):
        return False

    df.at[row_idx, "Cassava ton"] = new_value
    if "Ethanol litres" in df.columns:
        df.at[row_idx, "Ethanol litres"] = new_value * ETHANOL_LITRES_PER_TON
    if "Animal Feed ton" in df.columns:
        df.at[row_idx, "Animal Feed ton"] = new_value * ANIMAL_FEED_TON_PER_TON

    table.set_data(df, mark_user_input=True)
    st.session_state[PRODUCTION_EDIT_FLAG] = True
    _mark_inputs_dirty()
    _update_table_editor_state(table)
    return True


def _render_production_panel(page: InputLandingPage) -> None:
    """Expose production schedules outside the grouped tab layout."""

    st.subheader("Production Schedule")
    st.markdown(
        """
        - **Cascade growth across the projection horizon.**
        - Set a monthly growth percentage to apply from this month onward; the model will recalculate every subsequent production month automatically.
        - Updates are applied immediately—use the section tables below for bulk edits or row management.
        - The production schedule you see here is derived from your changes, not the seeded defaults.
        - Adjust cassava tonnage with the ± controls or insert dated changes on the right. Monthly figures automatically cascade to the annual production summary.
        """
    )

    monthly_table = page.production_monthly
    annual_table = page.production_annual

    monthly_df = monthly_table.data.copy()

    change_applied = False
    if not monthly_df.empty:
        month_col = _get_month_column(monthly_df) or monthly_table.columns[0]

        def _format_row(idx: int) -> str:
            if month_col in monthly_df.columns:
                value = monthly_df.at[idx, month_col]
                if pd.notna(value):
                    return str(value)
            return f"Row {idx + 1}"

        row_idx = st.selectbox(
            "Month to adjust",
            list(monthly_df.index),
            format_func=_format_row,
            key="production_month_select",
        )

        step_value = st.number_input(
            "Adjustment step (tons)",
            min_value=0.0,
            value=100.0,
            step=10.0,
            key="production_step_value",
        )

        growth_col = next((c for c in monthly_df.columns if "growth" in c.lower()), None)
        default_growth_pct = 0.0
        if growth_col is not None and row_idx in monthly_df.index:
            current_growth = pd.to_numeric(
                pd.Series([monthly_df.at[row_idx, growth_col]]), errors="coerce"
            ).iloc[0]
            if pd.notna(current_growth):
                default_growth_pct = float(current_growth) * 100.0

        controls = st.columns([1, 1, 2, 2])
        minus_pressed = controls[0].button("−", key=f"production_minus_{row_idx}")
        plus_pressed = controls[1].button("+", key=f"production_plus_{row_idx}")

        with controls[2]:
            st.markdown("**Monthly growth % to apply**")
            growth_value = st.number_input(
                "Monthly growth % to apply",
                value=float(default_growth_pct),
                step=0.1,
                format="%.2f",
                key=f"production_growth_pct_{row_idx}",
            )
        with controls[3]:
            if st.button(
                "Cascade growth",
                key=f"production_cascade_growth_{row_idx}",
                help="Apply the selected growth rate from this month onward.",
            ):
                _apply_growth_cascade(
                    page,
                    monthly_table,
                    monthly_table.data.copy(),
                    row_idx,
                    growth_value,
                )
                change_applied = True

        st.markdown("**Change effective month**")
        st.caption(
            "Select the month where the new production plan should begin, click **Add change**, and edit the inserted row. "
            "The model will cascade cassava tonnage (and the derived ethanol and animal feed outputs) to all later months automatically."
        )
        change_controls_applied = _render_change_controls(page, monthly_table)
        change_applied = change_applied or change_controls_applied

        if step_value > 0:
            if minus_pressed:
                _apply_production_delta(page, monthly_table, row_idx, -step_value)
            if plus_pressed:
                _apply_production_delta(page, monthly_table, row_idx, step_value)
        elif minus_pressed or plus_pressed:
            st.warning("Set a step above zero to apply the adjustment.")

    else:
        st.info(
            "The production schedule is empty. Use the bulk change controls to insert the first production month."
        )
        st.markdown("**Change effective month**")
        st.caption(
            "Select the month where the new production plan should begin, click **Add change**, and edit the newly inserted row. "
            "The model will cascade cassava tonnage (and the derived ethanol and animal feed outputs) to all later months automatically."
        )
        change_applied = _render_change_controls(page, monthly_table)

    if change_applied and monthly_table.name == "Production Monthly":
        st.session_state[PRODUCTION_EDIT_FLAG] = True

    _render_table(
        page,
        monthly_table,
        "Production",
        expanded=True,
        allow_change_controls=False,
    )

    _render_table(
        page,
        annual_table,
        "Production",
        expanded=False,
        allow_change_controls=False,
    )


def _editable_tables(page: InputLandingPage) -> None:
    """Render editable data tables grouped by the landing-page sections."""

    categories = page.grouped_tables()
    filtered = OrderedDict((k, v) for k, v in categories.items() if k != "Production")
    if not filtered:
        return

    tabs = st.tabs(list(filtered.keys()))

    for tab, (section, tables) in zip(tabs, filtered.items()):
        with tab:
            for table in tables:
                expanded = section in {"Global", "Capex", "Financial"}
                _render_table(page, table, section, expanded=expanded)
                if table is page.initial_investment:
                    st.metric("Total Initial Investment", _format_currency(page.total_initial_investment))


def _render_table(
    page: InputLandingPage,
    table: EditableTable,
    section: str,
    *,
    expanded: bool = False,
    allow_change_controls: bool = True,
) -> None:
    """Show a Streamlit data editor for a specific table."""

    safe_key = table.name.replace(' ', '_').lower()
    widget_key = _table_widget_key(table)
    cache_key = _table_editor_state_key(table)
    with st.expander(table.name, expanded=expanded):
        original_data = table.data.copy()
        data_changed = False
        controls = st.columns(2)
        if controls[0].button("➕ Add row", key=f"add_{safe_key}"):
            table.add_row({column: None for column in table.columns})
            data_changed = True

        if not table.data.empty:
            row_options = list(table.data.index)
            remove_index = controls[1].selectbox(
                "Row to remove",
                options=row_options,
                key=f"remove_select_{safe_key}",
                label_visibility="collapsed",
                format_func=lambda i: f"Row {i + 1}",
            )
            if controls[1].button("➖ Remove selected", key=f"remove_{safe_key}"):
                idx = int(remove_index)
                if (
                    table.name == "Direct Costs Monthly"
                    and "Cost Category" in table.data.columns
                    and _is_cassava_feedstock(table.data.at[idx, "Cost Category"])
                ):
                    st.warning("Cassava feedstock costs are scenario-driven and cannot be removed.")
                else:
                    table.remove_row(idx)
                    data_changed = True
        else:
            controls[1].markdown("&nbsp;")

        change_applied = False
        if allow_change_controls:
            change_applied = _render_change_controls(page, table)
            if change_applied:
                data_changed = True

        derived_columns = DERIVED_COLUMN_MAP.get(table.name, set())
        column_config = {}
        for col in derived_columns:
            if col in table.columns:
                column_config[col] = st.column_config.NumberColumn(
                    label=col,
                    disabled=True,
                    help="Calculated automatically from other inputs.",
                )

        if table.name == "Production Monthly":
            st.caption(
                "Edit **Cassava ton** (and optional Growth %) for any month. The model will automatically recompute the matching "
                "ethanol litres and animal-feed tonnage and roll the results into the annual production table."
            )
        elif table.name == "Production Annual":
            st.caption(
                "These annual totals are derived from the monthly production schedule. Update the monthly table to change the "
                "values shown here."
            )

        if cache_key not in st.session_state:
            st.session_state[cache_key] = table.data.copy()

        if table.name == "Direct Costs Monthly" and "Cost Category" in table.data.columns:
            auto_mask = table.data["Cost Category"].apply(_is_cassava_feedstock)
            auto_rows = table.data.loc[auto_mask].copy()
            manual_rows = table.data.loc[~auto_mask].copy()

            if not auto_rows.empty:
                auto_rows = auto_rows.sort_index()
                st.caption(
                    "Cassava feedstock costs are scenario-driven and locked. Adjust other "
                    "direct-cost rows using the editor below."
                )
                st.data_editor(
                    auto_rows,
                    use_container_width=True,
                    key=f"{widget_key}_auto",
                    disabled=True,
                    column_config=column_config or None,
                )

            if manual_rows.empty:
                st.info("No editable direct-cost rows are available. Use **Add row** to insert a new item.")
                manual_result = manual_rows.copy()
            else:
                manual_column_config = dict(column_config)
                if "Cost Category" in manual_rows.columns:
                    manual_column_config["Cost Category"] = st.column_config.SelectboxColumn(
                        label="Cost Category",
                        options=list(DIRECT_COST_CATEGORY_OPTIONS),
                    )
                edited_manual = st.data_editor(
                    manual_rows,
                    num_rows="dynamic",
                    use_container_width=True,
                    key=f"{widget_key}_manual",
                    column_config=manual_column_config or None,
                )
                if isinstance(edited_manual, pd.DataFrame):
                    manual_result = edited_manual[manual_rows.columns].copy()
                else:  # pragma: no cover
                    manual_result = pd.DataFrame(edited_manual, columns=manual_rows.columns)
                if not manual_result.equals(manual_rows):
                    data_changed = True

            combined = pd.concat([auto_rows, manual_result], axis=0).reindex(table.data.index)
            new_data = combined[table.columns]
        else:
            edited = st.data_editor(
                table.data,
                num_rows="dynamic",
                use_container_width=True,
                key=widget_key,
                column_config=column_config or None,
            )
            if isinstance(edited, pd.DataFrame):
                new_data = edited[table.columns].copy()
            else:  # pragma: no cover - safety for older Streamlit returning list of dicts
                new_data = pd.DataFrame(edited, columns=table.columns)

            if not new_data.equals(table.data):
                data_changed = True

        table.set_data(new_data, mark_user_input=data_changed)
        if (data_changed or change_applied) and table.name == "Production Monthly":
            st.session_state[PRODUCTION_EDIT_FLAG] = True
        _update_table_editor_state(table)

        if data_changed or not table.data.equals(original_data):
            _mark_inputs_dirty()

def _annualise(rate: float | None) -> float | None:
    if rate is None or pd.isna(rate):
        return None
    return (1 + rate) ** 12 - 1


def _format_rate(rate: float | None) -> str:
    if rate is None or pd.isna(rate):
        return "n/a"
    return f"{rate * 100:,.2f}%"


def _format_percent(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value * 100:,.1f}%"


def _format_currency(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"${value:,.0f}"


def _reset_period_index(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=[label])
    result = df.copy().reset_index()
    if result.columns[0] != label:
        result = result.rename(columns={result.columns[0]: label})
    if np.issubdtype(result[label].dtype, np.datetime64):
        result[label] = pd.to_datetime(result[label]).dt.to_period("M").astype(str)
    return result

def _render_key_metrics(model: CassavaBioethanolModel, results: Dict[str, object]) -> None:
    metrics = results["metrics"]
    revenue = results["revenue"]
    production = results["production"]
    costs = results["costs"]
    expenses_summary: ExpenseSummary | None = results.get("expenses") if isinstance(results.get("expenses"), ExpenseSummary) else None
    financials = results["financials"]
    loan_schedule = results["loan_schedule"]
    depreciation = results["depreciation"]

    st.subheader("Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Project NPV", _format_currency(metrics.get("Project NPV")))
    col2.metric("Project IRR (annual)", _format_rate(_annualise(metrics.get("Project IRR"))))
    col3.metric("Equity IRR (annual)", _format_rate(_annualise(metrics.get("Equity IRR"))))
    payback_years = metrics.get("Payback Period (years)")
    col4.metric("Payback Period (years)", f"{payback_years:,.1f}" if payback_years and not pd.isna(payback_years) else "n/a")

    st.markdown("### Assumptions Snapshot")
    assumption_snapshot = pd.DataFrame(
        {
            "Assumption": [
                "Corporate Tax Rate",
                "Investor Share",
                "Owner Share",
                "Discount Rate",
                "Terminal Growth Rate",
                "Capital Gains Tax Rate",
                "Total Initial Investment",
            ],
            "Value": [
                _format_rate(metrics.get("Corporate Tax Rate")),
                _format_percent(metrics.get("Investor Share")),
                _format_percent(metrics.get("Owner Share")),
                _format_rate(metrics.get("Discount Rate")),
                _format_rate(metrics.get("Terminal Growth Rate")),
                _format_rate(metrics.get("Capital Gains Tax Rate")),
                _format_currency(metrics.get("Total Initial Investment")),
            ],
        }
    )
    st.dataframe(assumption_snapshot, use_container_width=True, hide_index=True)

    st.markdown("### Latest Drivers")
    drivers = pd.DataFrame(
        {
            "Metric": [
                "Final Month Revenue",
                "Final Month EBITDA",
                "Final Month Equity Cash Flow",
                "Cumulative FCF",
                "Cumulative Equity CF",
            ],
            "Value": [
                _format_currency(metrics.get("Final Month Revenue")),
                _format_currency(metrics.get("Final Month EBITDA")),
                _format_currency(metrics.get("Final Month Equity CF")),
                _format_currency(metrics.get("Cumulative FCF")),
                _format_currency(metrics.get("Cumulative Equity CF")),
            ],
        }
    )
    st.dataframe(drivers, use_container_width=True, hide_index=True)

    st.markdown("### Annual Operations & Production")
    production_annual = production.annual.copy()
    if not production_annual.empty:
        chart_data = production_annual.select_dtypes(include=[np.number])
        if not chart_data.empty:
            st.bar_chart(chart_data)
        else:
            st.info("No numeric production data available for charting.")
    else:
        st.info("No production data available for the selected horizon.")

    summary_cols = [col for col in ["Revenue", "EBITDA", "Net Income"] if col in financials.income_annual.columns]
    if summary_cols:
        annual_summary = financials.income_annual[summary_cols].copy()
        annual_summary.index.name = "Year"
        st.dataframe(annual_summary.reset_index(), use_container_width=True)

    st.markdown("### Cash Flow & Returns")
    cash_columns = [
        c
        for c in [
            "Operating Cash Flow",
            "Investing Cash Flow",
            "Financing Cash Flow",
            "Free Cash Flow",
            "Equity Cash Flow",
        ]
        if c in financials.cashflow_monthly.columns
    ]
    if cash_columns:
        st.line_chart(financials.cashflow_monthly[cash_columns])
        cumulative_df = financials.cashflow_monthly[cash_columns].cumsum()
        cumulative_df.columns = [f"Cumulative {col}" for col in cumulative_df.columns]
        st.line_chart(cumulative_df)

    st.markdown("### Revenue Mix")
    revenue_annual = revenue.annual.copy()
    if not revenue_annual.empty:
        if "Total Revenue" in revenue_annual:
            mix_df = revenue_annual.drop(columns=["Total Revenue"])
        else:
            mix_df = revenue_annual
        if not mix_df.empty:
            st.bar_chart(mix_df)
        st.dataframe(revenue_annual.reset_index().rename(columns={"index": "Year"}), use_container_width=True)
    else:
        st.info("Revenue inputs are empty for the current projection.")

    st.markdown("### Operating Cost Breakdown")
    if isinstance(expenses_summary, ExpenseSummary) and not expenses_summary.monthly.empty:
        st.area_chart(expenses_summary.monthly)
    else:
        cost_monthly = pd.DataFrame(
            {
                name: output.monthly.sum(axis=1)
                for name, output in costs.items()
                if output and not output.monthly.empty
            }
        )
        if not cost_monthly.empty:
            st.area_chart(cost_monthly)

    if isinstance(expenses_summary, ExpenseSummary) and not expenses_summary.annual.empty:
        annual_expense = expenses_summary.annual.copy()
        annual_expense.index.name = "Year"
        st.dataframe(annual_expense.reset_index(), use_container_width=True)
    else:
        cost_annual = pd.DataFrame(
            {
                name: output.annual.sum(axis=1)
                for name, output in costs.items()
                if output and not output.annual.empty
            }
        )
        if not cost_annual.empty:
            cost_annual.index.name = "Year"
            st.dataframe(cost_annual.reset_index(), use_container_width=True)

    st.markdown("### Capital Expenditure & Debt")
    capex_df = model.input_page.initial_investment.model_frame
    if not capex_df.empty:
        st.bar_chart(capex_df.set_index("Item")["Cost"])
        st.dataframe(capex_df, use_container_width=True)
    debt_chart = loan_schedule.schedule.pivot_table(index="Month", values="Closing Balance", aggfunc="sum")
    if not debt_chart.empty:
        st.line_chart(debt_chart)

    st.markdown("### Fixed Asset Summary")
    st.dataframe(depreciation.summary, use_container_width=True)

    st.markdown("### Debt Schedule Chart")
    debt_payments = loan_schedule.schedule.pivot_table(index="Month", values="Payment", aggfunc="sum")
    if not debt_payments.empty:
        st.area_chart(debt_payments)

    st.markdown("### Break-even Analysis")
    break_even_df = results.get("break_even")
    if isinstance(break_even_df, pd.DataFrame) and not break_even_df.empty:
        st.dataframe(_reset_period_index(break_even_df, "Month"), use_container_width=True)

def _render_financial_performance(results: Dict[str, object]) -> None:
    financials = results["financials"]
    expenses_summary: ExpenseSummary | None = (
        results.get("expenses") if isinstance(results.get("expenses"), ExpenseSummary) else None
    )
    costs = results["costs"]

    st.subheader("Monthly Financial Performance")
    st.dataframe(_reset_period_index(financials.income_monthly, "Month"), use_container_width=True)

    st.subheader("Annual Financial Performance")
    annual_income = financials.income_annual.copy()
    annual_income.index.name = "Year"
    st.dataframe(annual_income.reset_index(), use_container_width=True)

    expense_columns = ["COGS", "Staff Costs", "Other Opex"]
    monthly_expense_breakdown = financials.income_monthly.reindex(columns=[c for c in expense_columns if c in financials.income_monthly.columns])
    if monthly_expense_breakdown.empty:
        monthly_expense_breakdown = pd.DataFrame(0.0, index=financials.income_monthly.index, columns=expense_columns)
    st.subheader("Expense Breakdown (Monthly)")
    st.dataframe(_reset_period_index(monthly_expense_breakdown, "Month"), use_container_width=True)

    annual_expense_breakdown = financials.income_annual.reindex(columns=[c for c in expense_columns if c in financials.income_annual.columns])
    if annual_expense_breakdown.empty:
        annual_expense_breakdown = pd.DataFrame(0.0, index=financials.income_annual.index, columns=expense_columns)
    annual_expense_breakdown = annual_expense_breakdown.copy()
    annual_expense_breakdown.index.name = "Year"
    st.subheader("Expense Breakdown (Annual)")
    st.dataframe(annual_expense_breakdown.reset_index(), use_container_width=True)

    income_ratios_monthly = getattr(financials, "income_ratios_monthly", pd.DataFrame())
    if isinstance(income_ratios_monthly, pd.DataFrame) and not income_ratios_monthly.empty:
        ratio_monthly = _reset_period_index(income_ratios_monthly, "Month")
        if "Month" in ratio_monthly.columns:
            try:
                ratio_monthly["Month"] = pd.to_datetime(ratio_monthly["Month"]).dt.to_period("M").astype(str)
            except Exception:
                ratio_monthly["Month"] = ratio_monthly["Month"].astype(str)
        st.subheader("Income Statement Ratios (Monthly)")
        st.dataframe(ratio_monthly, use_container_width=True)

    income_ratios_annual = getattr(financials, "income_ratios_annual", pd.DataFrame())
    if isinstance(income_ratios_annual, pd.DataFrame) and not income_ratios_annual.empty:
        ratio_annual = _reset_period_index(income_ratios_annual, "Year")
        st.subheader("Income Statement Ratios (Annual)")
        st.dataframe(ratio_annual, use_container_width=True)

    st.subheader("Total Expense Schedule")
    if isinstance(expenses_summary, ExpenseSummary):
        if not expenses_summary.monthly.empty:
            st.dataframe(
                _reset_period_index(expenses_summary.monthly, "Month"),
                use_container_width=True,
            )
        if not expenses_summary.annual.empty:
            annual_expense = expenses_summary.annual.copy()
            annual_expense.index.name = "Year"
            st.dataframe(annual_expense.reset_index(), use_container_width=True)
    else:
        monthly_expense = pd.DataFrame(
            {
                name: output.monthly.sum(axis=1)
                for name, output in costs.items()
                if output and not output.monthly.empty
            }
        )
        if not monthly_expense.empty:
            st.dataframe(_reset_period_index(monthly_expense, "Month"), use_container_width=True)
        annual_expense = pd.DataFrame(
            {
                name: output.annual.sum(axis=1)
                for name, output in costs.items()
                if output and not output.annual.empty
            }
        )
        if not annual_expense.empty:
            annual_expense.index.name = "Year"
            st.dataframe(annual_expense.reset_index(), use_container_width=True)

    staff_schedule = results.get("staff_schedule")
    if staff_schedule is not None:
        positions_df = getattr(staff_schedule, "positions", pd.DataFrame())
        summary_df = getattr(staff_schedule, "department_summary", pd.DataFrame())
        if isinstance(positions_df, pd.DataFrame) and not positions_df.empty:
            st.subheader("Staff Position Schedule")
            st.dataframe(positions_df, use_container_width=True)
        if isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
            st.subheader("Staff Cost by Department")
            st.dataframe(summary_df, use_container_width=True)

def _render_financial_position(results: Dict[str, object]) -> None:
    financials = results["financials"]

    st.subheader("Monthly Statement of Financial Position")
    st.dataframe(_reset_period_index(financials.balance_monthly, "Month"), use_container_width=True)

    st.subheader("Annual Statement of Financial Position")
    balance_annual = financials.balance_annual.copy()
    balance_annual.index.name = "Year"
    st.dataframe(balance_annual.reset_index(), use_container_width=True)

    balance_ratios_monthly = getattr(financials, "balance_ratios_monthly", pd.DataFrame())
    if isinstance(balance_ratios_monthly, pd.DataFrame) and not balance_ratios_monthly.empty:
        ratio_monthly = _reset_period_index(balance_ratios_monthly, "Month")
        if "Month" in ratio_monthly.columns:
            try:
                ratio_monthly["Month"] = pd.to_datetime(ratio_monthly["Month"]).dt.to_period("M").astype(str)
            except Exception:
                ratio_monthly["Month"] = ratio_monthly["Month"].astype(str)
        st.subheader("Statement of Financial Position Ratios (Monthly)")
        st.dataframe(ratio_monthly, use_container_width=True)

    balance_ratios_annual = getattr(financials, "balance_ratios_annual", pd.DataFrame())
    if isinstance(balance_ratios_annual, pd.DataFrame) and not balance_ratios_annual.empty:
        ratio_annual = _reset_period_index(balance_ratios_annual, "Year")
        st.subheader("Statement of Financial Position Ratios (Annual)")
        st.dataframe(ratio_annual, use_container_width=True)

def _render_cash_flow_page(results: Dict[str, object]) -> None:
    financials = results["financials"]

    st.subheader("Monthly Cash Flow Statement")
    cash_monthly = financials.cashflow_monthly
    st.dataframe(_reset_period_index(cash_monthly, "Month"), use_container_width=True)

    st.subheader("Annual Cash Flow Statement")
    cash_annual = financials.cashflow_annual.copy()
    cash_annual.index.name = "Year"
    st.dataframe(cash_annual.reset_index(), use_container_width=True)

    st.subheader("Cash Flow and Returns Charts")
    cash_columns = [
        c
        for c in [
            "Operating Cash Flow",
            "Investing Cash Flow",
            "Financing Cash Flow",
            "Free Cash Flow",
            "Equity Cash Flow",
        ]
        if c in cash_monthly.columns
    ]
    if cash_columns:
        st.line_chart(cash_monthly[cash_columns])

    st.subheader("Cumulative Equity Cash Flow")
    if "Equity Cash Flow" in cash_monthly.columns:
        cumulative_equity = cash_monthly[["Equity Cash Flow"]].cumsum()
        cumulative_equity.columns = ["Cumulative Equity Cash Flow"]
        st.line_chart(cumulative_equity)
        cumulative_df = cumulative_equity.reset_index().rename(columns={cumulative_equity.index.name or "index": "Month"})
        st.dataframe(cumulative_df, use_container_width=True)

def _render_sensitivity_page(model: CassavaBioethanolModel, results: Dict[str, object]) -> None:
    st.subheader("Sensitivity Analysis Configuration")
    config_df = pd.DataFrame([s.__dict__ for s in DEFAULT_SENSITIVITY_SCENARIOS]) if DEFAULT_SENSITIVITY_SCENARIOS else pd.DataFrame(columns=["name", "parameter", "delta"])
    st.dataframe(config_df.rename(columns={"name": "Scenario", "parameter": "Parameter", "delta": "Delta"}), use_container_width=True, hide_index=True)

    base_page = copy.deepcopy(results.get("input_page_snapshot", model.input_page))

    def _scenario_model() -> CassavaBioethanolModel:
        clone = CassavaBioethanolModel(copy.deepcopy(base_page))
        clone.scenario = model.scenario
        return clone

    analysis_model = _scenario_model()
    cache: Dict[str, Dict[str, object]] = st.session_state.setdefault(SENSITIVITY_CACHE_KEY, {})
    cached_entry = cache.get(model.scenario)
    run_requested = st.button(
        "Run Sensitivity Analysis",
        key=f"run_sensitivity_{model.scenario.lower()}",
    )

    if run_requested or not cached_entry or cached_entry.get("version") != _current_model_version():
        if DEFAULT_SENSITIVITY_SCENARIOS:
            with st.spinner("Running sensitivity cases..."):
                sensitivity_results = run_sensitivity(analysis_model, DEFAULT_SENSITIVITY_SCENARIOS)
        else:
            sensitivity_results = pd.DataFrame(
                columns=["Scenario", "Parameter", "Delta", "Project NPV", "Change vs Base"]
            )
        cache[model.scenario] = {
            "version": _current_model_version(),
            "results": sensitivity_results,
        }
        st.session_state[SENSITIVITY_CACHE_KEY] = cache
        cached_entry = cache[model.scenario]
    elif cached_entry is None:
        sensitivity_results = pd.DataFrame(
            columns=["Scenario", "Parameter", "Delta", "Project NPV", "Change vs Base"]
        )
    else:
        sensitivity_results = cached_entry.get("results", pd.DataFrame())

    st.subheader("Simulation Results")
    if sensitivity_results.empty:
        st.info("Click 'Run Sensitivity Analysis' to generate comparison results.")
    else:
        st.dataframe(sensitivity_results, use_container_width=True)

    st.subheader("Monte Carlo Simulation Configuration")
    mc_rows = (
        [{"Setting": "Iterations", "Value": MONTE_CARLO_ITERATIONS}, {"Setting": "Random Seed", "Value": MONTE_CARLO_SEED}]
        + [{"Setting": f"Std Dev - {param}", "Value": std} for param, std in MONTE_CARLO_STD.items()]
    )
    st.dataframe(pd.DataFrame(mc_rows), use_container_width=True, hide_index=True)

    st.subheader("Tornado Drivers")
    tornado_model = _scenario_model()
    tornado_df = tornado_chart_inputs(tornado_model, TORNADO_DRIVERS, scale=0.1)
    st.dataframe(tornado_df, use_container_width=True)

def _render_scenario_page(model: CassavaBioethanolModel, results: Dict[str, object]) -> None:
    st.subheader("Scenario/Is Configuration")
    scenario_df = pd.DataFrame([{"Scenario": cfg.name, **cfg.overrides} for cfg in DEFAULT_SCENARIO_CONFIGS]) if DEFAULT_SCENARIO_CONFIGS else pd.DataFrame(columns=["Scenario"])
    st.dataframe(scenario_df, use_container_width=True)

    st.subheader("Scenario Tool Configuration")
    tool_df = model.input_page.global_inputs.model_frame.rename(columns={"Value": "Base Value"}).copy()
    numeric_values = pd.to_numeric(tool_df["Base Value"], errors="coerce")
    tool_df["Low Bound"] = np.where(numeric_values.notna(), numeric_values * 0.8, np.nan)
    tool_df["High Bound"] = np.where(numeric_values.notna(), numeric_values * 1.2, np.nan)
    desired_order = ["Parameter", "Base Value", "Units", "Low Bound", "High Bound"]
    tool_df = tool_df[[c for c in desired_order if c in tool_df.columns]]
    st.dataframe(tool_df, use_container_width=True, hide_index=True)

    base_page = copy.deepcopy(results.get("input_page_snapshot", model.input_page))

    def _scenario_model() -> CassavaBioethanolModel:
        clone = CassavaBioethanolModel(copy.deepcopy(base_page))
        clone.scenario = model.scenario
        return clone

    comparison_model = _scenario_model()
    scenario_cache: Dict[str, Dict[str, object]] = st.session_state.setdefault(SCENARIO_CACHE_KEY, {})
    cached_entry = scenario_cache.get(model.scenario)
    comparison_button = st.button(
        "Run Scenario Comparison",
        key=f"run_scenario_cmp_{model.scenario.lower()}",
    )

    if comparison_button or not cached_entry or cached_entry.get("version") != _current_model_version():
        if DEFAULT_SCENARIO_CONFIGS:
            with st.spinner("Evaluating scenario overrides..."):
                comparison_df = scenario_comparison(comparison_model, DEFAULT_SCENARIO_CONFIGS)
        else:
            comparison_df = pd.DataFrame(columns=["Scenario", "Project NPV", "Project IRR", "Equity IRR"])
        scenario_cache[model.scenario] = {
            "version": _current_model_version(),
            "comparison": comparison_df,
        }
        st.session_state[SCENARIO_CACHE_KEY] = scenario_cache
        cached_entry = scenario_cache[model.scenario]
    else:
        comparison_df = (
            cached_entry.get("comparison") if cached_entry else pd.DataFrame(columns=["Scenario", "Project NPV", "Project IRR", "Equity IRR"])
        )
    st.subheader("Scenario Comparison")
    if comparison_df.empty:
        st.info("Click 'Run Scenario Comparison' to evaluate the configured overrides.")
    else:
        st.dataframe(comparison_df, use_container_width=True)

    st.subheader("Goal Seek Results")
    goal_seek_parameter = "Corporate tax rate"
    goal_seek_metric = "Project NPV"
    goal_message: str | None = None
    empty_goal_df = pd.DataFrame(
        columns=[
            "Parameter",
            "Target Metric",
            "Target Value",
            "Target Name",
            "Achieved Value",
            "Tolerance",
            "Iterations",
        ]
    )

    try:
        target_value = float(results["metrics"].get(goal_seek_metric, 0.0))
        goal_model = _scenario_model()
        goal_result = goal_seek_to_target(goal_model, goal_seek_parameter, goal_seek_metric, target_value)
        goal_df = pd.DataFrame(
            [
                {
                    "Parameter": goal_seek_parameter,
                    "Target Metric": goal_seek_metric,
                    "Target Value": target_value,
                    "Target Name": goal_result.target_name,
                    "Achieved Value": goal_result.achieved_value,
                    "Tolerance": goal_result.tolerance,
                    "Iterations": goal_result.iterations,
                }
            ]
        )
    except KeyError:
        goal_df = empty_goal_df
        goal_message = "The selected goal seek parameter is not available in the current global inputs."
    except ValueError as exc:
        goal_df = empty_goal_df
        goal_message = str(exc)

    if goal_message:
        st.info(goal_message)

    st.dataframe(goal_df, use_container_width=True)

def _render_monte_carlo_page(model: CassavaBioethanolModel, results: Dict[str, object]) -> None:
    st.subheader("Monte Carlo Simulation")

    current_version = _current_model_version()
    current_scenario = model.scenario
    cache: Dict[str, Dict[str, object]] = st.session_state.setdefault(MC_CACHE_KEY, {})
    cached_entry = cache.get(current_scenario)

    run_requested = st.button(
        "Run Monte Carlo Simulation",
        key=f"mc_run_{current_scenario.lower()}",
    )

    if run_requested or not cached_entry or cached_entry.get("version") != current_version:
        if run_requested:
            base_page = copy.deepcopy(results.get("input_page_snapshot", model.input_page))
            with st.spinner("Running Monte Carlo simulation..."):
                mc_model = CassavaBioethanolModel(copy.deepcopy(base_page))
                mc_model.scenario = current_scenario
                mc_results = monte_carlo_simulation(
                    mc_model,
                    parameter_std=MONTE_CARLO_STD,
                    iterations=MONTE_CARLO_ITERATIONS,
                    random_seed=MONTE_CARLO_SEED,
                )
            cache[current_scenario] = {"version": current_version, "results": mc_results}
            st.session_state[MC_CACHE_KEY] = cache
            cached_entry = cache[current_scenario]
        else:
            st.info("Click 'Run Monte Carlo Simulation' to generate results for this scenario.")
            return

    if not cached_entry or cached_entry.get("results") is None:
        st.info("Monte Carlo results are not available. Click the run button to generate them.")
        return

    mc_results = cached_entry["results"]
    if mc_results.empty:
        st.info("Monte Carlo results are not available for the current configuration.")
        return

    st.subheader("Summary Statistics")
    summary = mc_results.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).T
    st.dataframe(summary, use_container_width=True)

    st.subheader("NPV Distribution (sorted path)")
    st.line_chart(mc_results["Project NPV"].sort_values().reset_index(drop=True))

    if "Project IRR" in mc_results:
        st.subheader("IRR Distribution (sorted path)")
        st.line_chart(mc_results["Project IRR"].sort_values().reset_index(drop=True))

def main() -> None:
    st.title("Cassava_Bioethanol Financial Model")
    st.caption("Adjust the assumptions, run the project finance model, and inspect the outputs across dedicated dashboards.")

    input_page = _load_session_inputs()
    _sync_projection_from_session(input_page)
    scenario_options = list(CassavaBioethanolModel.SCENARIOS)
    if "selected_scenario" not in st.session_state:
        st.session_state.selected_scenario = scenario_options[0]

    action_cols = st.columns([1, 1, 1])
    with action_cols[0]:
        recalc = st.button("Recalculate model", type="primary")
    with action_cols[1]:
        scenario_index = scenario_options.index(st.session_state.selected_scenario)
        selected_choice = st.selectbox(
            "Scenario",
            scenario_options,
            index=scenario_index,
            key="scenario_select",
        )
    download_container = action_cols[2].container()

    if selected_choice != st.session_state.selected_scenario:
        st.session_state.selected_scenario = selected_choice
        st.session_state.scenario_payloads = {}
        st.session_state.excel_bytes_map = {}
        st.session_state[MC_CACHE_KEY] = {}
        st.session_state.pop("mc_cache", None)
        st.session_state.pop("mc_cache_version", None)
        st.session_state.pop("mc_cache_scenario", None)
        st.session_state[SENSITIVITY_CACHE_KEY] = {}
        st.session_state[SCENARIO_CACHE_KEY] = {}

    selected_scenario = st.session_state.selected_scenario

    st.caption("Use the navigation tabs to move between the input landing page and the analytical dashboards.")

    tabs = st.tabs(
        [
            "Input Landing Page",
            "Key Metrics Dashboard",
            "Financial Performance",
            "Financial Position",
            "Cash Flow Statement",
            "Sensitivity Analyses",
            "Scenario / IFs Analysis",
            "Monte Carlo Simulation",
        ]
    )

    with tabs[0]:
        st.subheader("Input Landing Page")
        st.info("Edit the assumptions and press 'Recalculate model' to refresh the financial outputs.")
        _update_projection(input_page)
        _key_assumptions_controls(input_page.global_inputs)
        _modify_default_inputs(input_page)
        _render_production_panel(input_page)
        _sync_table_editors(input_page)
        _apply_dependent_updates(input_page, selected_scenario)
        _editable_tables(input_page)

    snapshot = st.session_state.get("input_snapshot")
    snapshot_projection = None
    if snapshot is not None:
        snapshot_projection = (
            int(snapshot.projection.start_year),
            int(snapshot.projection.end_year),
            str(snapshot.projection.planning_start),
        )
    current_projection = (
        int(input_page.projection.start_year),
        int(input_page.projection.end_year),
        str(input_page.projection.planning_start),
    )
    projection_changed = snapshot_projection is not None and snapshot_projection != current_projection
    inputs_dirty = st.session_state.get("inputs_dirty", False)

    if (
        recalc
        or projection_changed
        or inputs_dirty
        or "scenario_payloads" not in st.session_state
    ):
        st.session_state.scenario_payloads = {}
        st.session_state.excel_bytes_map = {}
        st.session_state.input_snapshot = copy.deepcopy(input_page)
        _bump_model_version()
        st.session_state[MC_CACHE_KEY] = {}
        st.session_state.pop("mc_cache", None)
        st.session_state.pop("mc_cache_version", None)
        st.session_state.pop("mc_cache_scenario", None)
        st.session_state[SENSITIVITY_CACHE_KEY] = {}
        st.session_state[SCENARIO_CACHE_KEY] = {}
        st.session_state.inputs_dirty = False

    snapshot = st.session_state.get("input_snapshot")
    if snapshot is None:
        snapshot = copy.deepcopy(input_page)
        st.session_state.input_snapshot = snapshot

    model, results = _ensure_scenario_payload(selected_scenario, snapshot)
    st.session_state.model_results = (model, results)

    excel_map: Dict[str, bytes] = st.session_state.setdefault("excel_bytes_map", {})
    excel_bytes = excel_map.get(selected_scenario)

    model.scenario = selected_scenario

    with download_container:
        if not excel_bytes:
            if st.button("Prepare Excel Model", key=f"prepare_excel_{selected_scenario.lower()}"):
                with st.spinner("Preparing Excel workbook..."):
                    excel_bytes = _generate_excel_bytes(model, results, selected_scenario)
                excel_map[selected_scenario] = excel_bytes
                st.session_state.excel_bytes_map = excel_map
        if excel_bytes:
            st.download_button(
                "Download Excel Model",
                data=excel_bytes,
                file_name="Cassava_Bioethanol_Financial_Model.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            if st.button(
                "Clear Prepared Excel",
                key=f"clear_excel_{selected_scenario.lower()}",
            ):
                excel_map.pop(selected_scenario, None)
                st.session_state.excel_bytes_map = excel_map
                excel_bytes = None
        if not excel_bytes:
            st.info("Click 'Prepare Excel Model' to generate the workbook for download.")

    with tabs[1]:
        _render_key_metrics(model, results)

    with tabs[2]:
        _render_financial_performance(results)

    with tabs[3]:
        _render_financial_position(results)

    with tabs[4]:
        _render_cash_flow_page(results)

    with tabs[5]:
        _render_sensitivity_page(model, results)

    with tabs[6]:
        _render_scenario_page(model, results)

    with tabs[7]:
        _render_monte_carlo_page(model, results)


if __name__ == "__main__":
    main()
