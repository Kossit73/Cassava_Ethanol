"""Advanced analytical helpers for the cassava bioethanol model.

This module exposes a collection of utilities that complement the core
financial modelling workflow.  The goal is to provide a single place that
demonstrates how to tap into a richer scientific-computing ecosystem when
performing sensitivity analysis, optimisation or visualisation around the
model outputs.

The helpers rely on a broad set of third-party libraries – ranging from
``scipy`` distributions through ``scikit-learn`` estimators to ``plotly``
visualisations.  The functions deliberately use the imported packages so
that callers can discover practical examples of how each dependency fits
into the modelling story.
"""

from __future__ import annotations

import base64
import logging
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from functools import wraps
from io import BytesIO
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy_financial as npf
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from fastapi import HTTPException

from .scipy_compat import (
    bernoulli,
    beta,
    binom,
    chi2,
    expon,
    f,
    gamma,
    geom,
    hypergeom,
    lognorm,
    minimize,
    multinomial,
    norm,
    poisson,
    stats,
    uniform,
    weibull_min,
)
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from statsmodels.tsa.arima.model import ARIMA
from xlsxwriter.exceptions import DuplicateWorksheetName

# ``signal`` is not available on every platform (e.g. Windows under WSL) but it
# is useful to support native alarm-based timeouts when possible.  We import it
# lazily so that the module still loads even when ``SIGALRM`` is missing.
try:  # pragma: no cover - platform specific behaviour
    import signal
except Exception:  # pragma: no cover - ``signal`` may be unavailable
    signal = None  # type: ignore

LOGGER = logging.getLogger(__name__)


def _ensure_numeric_frame(df: pd.DataFrame, target: str) -> tuple[np.ndarray, np.ndarray, List[str]]:
    """Return ``(X, y, feature_names)`` for regression style problems.

    The helper converts non-numeric columns using one-hot encoding and drops
    rows with missing targets to keep the downstream estimators stable.
    """

    if target not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target}' not present in data frame")

    filtered = df.dropna(subset=[target])
    features = filtered.drop(columns=[target])
    encoded = pd.get_dummies(features, drop_first=True)
    if encoded.empty:
        raise HTTPException(status_code=400, detail="No explanatory features available after encoding")
    feature_names = list(encoded.columns)
    X = encoded.to_numpy(dtype=float)
    y = filtered[target].to_numpy(dtype=float)
    return X, y, feature_names


def run_with_timeout(timeout_seconds: float):
    """Decorator executing the wrapped callable with a timeout.

    We prefer using a ``ThreadPoolExecutor`` because it works on every
    platform.  When possible a supplementary ``signal`` based alarm is used to
    unblock the thread, ensuring that CPU intensive tasks do not run
    indefinitely during interactive sessions.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            LOGGER.debug("Running %s with timeout=%s", func.__name__, timeout_seconds)
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)

                def _signal_handler(signum, frame):  # pragma: no cover - platform specific
                    raise TimeoutError()

                alarm_enabled = False
                previous_handler = None
                if signal is not None and threading.current_thread() is threading.main_thread():
                    try:  # pragma: no cover - platform specific
                        previous_handler = signal.signal(signal.SIGALRM, _signal_handler)
                        signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
                        alarm_enabled = True
                    except Exception:  # pragma: no cover - signal may be unsupported
                        alarm_enabled = False

                try:
                    return future.result(timeout=timeout_seconds)
                except TimeoutError as exc:
                    future.cancel()
                    LOGGER.warning("Function %s exceeded timeout", func.__name__)
                    raise HTTPException(status_code=504, detail="Operation timed out") from exc
                finally:
                    if alarm_enabled and signal is not None:  # pragma: no cover - platform specific
                        signal.setitimer(signal.ITIMER_REAL, 0)
                        if previous_handler is not None:
                            signal.signal(signal.SIGALRM, previous_handler)

        return wrapper

    return decorator


@dataclass
class RegressionResult:
    coefficients: Dict[str, float]
    intercept: float
    score: float


@dataclass
class DecisionTreeResult:
    feature_importances: Dict[str, float]
    depth: int
    score: float


class AdvancedAnalyticsToolkit:
    """Bundle of utilities that expose advanced modelling techniques."""

    def __init__(self, model: "CassavaBioethanolModel") -> None:
        self.model = model
        self._scaler = StandardScaler()
        self._rng = np.random.default_rng()
        LOGGER.debug("AdvancedAnalyticsToolkit initialised for scenario: %s", getattr(model, "scenario", "unknown"))

    # ------------------------------------------------------------------
    # Regression utilities
    # ------------------------------------------------------------------
    def linear_regression(self, df: pd.DataFrame, target: str) -> RegressionResult:
        """Fit a linear regression model to ``df`` and return coefficients."""

        X, y, feature_names = _ensure_numeric_frame(df, target)
        scaled = self._scaler.fit_transform(X) if X.size else X
        reg = LinearRegression()
        reg.fit(scaled, y)
        return RegressionResult(
            coefficients={name: float(coeff) for name, coeff in zip(feature_names, reg.coef_)},
            intercept=float(reg.intercept_),
            score=float(reg.score(scaled, y)),
        )

    def neural_network_regression(
        self,
        df: pd.DataFrame,
        target: str,
        hidden_layer_sizes: Sequence[int] | None = None,
        random_state: int | None = 0,
    ) -> RegressionResult:
        """Fit a small feed-forward neural network on the provided data."""

        X, y, feature_names = _ensure_numeric_frame(df, target)
        scaled = self._scaler.fit_transform(X) if X.size else X
        hidden = tuple(hidden_layer_sizes) if hidden_layer_sizes else (64, 32)
        mlp = MLPRegressor(hidden_layer_sizes=hidden, activation="relu", random_state=random_state, max_iter=2000)
        mlp.fit(scaled, y)
        score = float(mlp.score(scaled, y)) if X.size else float("nan")
        weights = mlp.coefs_[0].mean(axis=1) if mlp.coefs_ else np.zeros(len(feature_names))
        return RegressionResult(
            coefficients={name: float(weight) for name, weight in zip(feature_names, weights)},
            intercept=float(mlp.intercepts_[0][0] if mlp.intercepts_ else 0.0),
            score=score,
        )

    def decision_tree_regression(
        self,
        df: pd.DataFrame,
        target: str,
        max_depth: int | None = None,
        random_state: int | None = 0,
    ) -> DecisionTreeResult:
        """Fit a decision tree regressor and return feature importances."""

        X, y, feature_names = _ensure_numeric_frame(df, target)
        tree = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
        tree.fit(X, y)
        score = float(tree.score(X, y)) if X.size else float("nan")
        importances = {name: float(val) for name, val in zip(feature_names, tree.feature_importances_)}
        return DecisionTreeResult(feature_importances=importances, depth=int(tree.get_depth()), score=score)

    # ------------------------------------------------------------------
    # Time-series forecasting
    # ------------------------------------------------------------------
    def arima_forecast(self, series: pd.Series, order: tuple[int, int, int] = (1, 1, 1), periods: int = 12) -> pd.Series:
        """Generate an ARIMA forecast for the supplied time-series."""

        if series.empty:
            raise HTTPException(status_code=400, detail="Cannot run ARIMA on an empty series")
        model = ARIMA(series.astype(float), order=order)
        fitted = model.fit()
        forecast = fitted.get_forecast(steps=periods)
        return forecast.predicted_mean

    def revolver_projection(self, series: pd.Series, window: int = 12) -> pd.DataFrame:
        """Return rolling statistics that approximate revolver utilisation."""

        if not isinstance(series, pd.Series) or series.empty:
            return pd.DataFrame(columns=["Value", "Rolling Mean", "Rolling Std"])

        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if numeric.empty:
            return pd.DataFrame(columns=["Value", "Rolling Mean", "Rolling Std"])

        window = max(int(window), 1)
        frame = pd.DataFrame({"Value": numeric})
        frame["Rolling Mean"] = frame["Value"].rolling(window=window, min_periods=1).mean()
        frame["Rolling Std"] = frame["Value"].rolling(window=window, min_periods=1).std().fillna(0.0)
        return frame

    # ------------------------------------------------------------------
    # Financial helpers
    # ------------------------------------------------------------------
    def discounted_cashflow_summary(self, rate: float) -> Dict[str, float]:
        """Return NPV, IRR and payback-style metrics for the model cash flows."""

        results = self.model.build()
        financials = results.get("financials")
        cashflow = getattr(financials, "cashflow_monthly", None)
        if cashflow is None or cashflow.empty:
            raise HTTPException(status_code=404, detail="Cashflow schedule is not available")

        project_cashflows = cashflow.get("Net Cash Flow", pd.Series(dtype=float)).to_numpy(dtype=float)
        discounted = npf.npv(rate, project_cashflows)
        irr_value = npf.irr(project_cashflows)

        cumulative = np.cumsum(project_cashflows)
        payback_period = float(np.argmax(cumulative > 0)) if np.any(cumulative > 0) else math.inf

        return {
            "discount_rate": float(rate),
            "net_present_value": float(discounted),
            "internal_rate_of_return": float(irr_value),
            "payback_period": payback_period,
        }

    # ------------------------------------------------------------------
    # Distribution analysis utilities
    # ------------------------------------------------------------------
    _DISTRIBUTIONS: Mapping[str, Any] = {
        "normal": norm,
        "lognormal": lognorm,
        "uniform": uniform,
        "exponential": expon,
        "bernoulli": bernoulli,
        "binomial": binom,
        "poisson": poisson,
        "geometric": geom,
        "chi_squared": chi2,
        "gamma": gamma,
        "weibull_min": weibull_min,
        "hypergeometric": hypergeom,
        "multinomial": multinomial,
        "beta": beta,
        "f": f,
    }

    def sample_distribution(self, name: str, size: int = 1000, **kwargs: Any) -> np.ndarray:
        """Sample random variates from one of the supported distributions."""

        key = name.lower()
        if key not in self._DISTRIBUTIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported distribution '{name}'")
        dist = self._DISTRIBUTIONS[key]

        kwargs.setdefault("random_state", self._rng)
        if dist is multinomial:
            n = int(kwargs.pop("n", 10))
            p = kwargs.pop("p", [0.5, 0.5])
            return dist.rvs(n, p, size=size, random_state=self._rng)
        if dist is binom:
            n = int(kwargs.pop("n", 10))
            p = float(kwargs.pop("p", 0.5))
            return dist.rvs(n, p, size=size, random_state=self._rng)
        if dist is poisson:
            mu = float(kwargs.pop("mu", kwargs.pop("lam", 5.0)))
            return dist.rvs(mu, size=size, random_state=self._rng)
        if dist is geom:
            p_val = float(kwargs.pop("p", 0.3))
            return dist.rvs(p_val, size=size, random_state=self._rng)
        if dist is bernoulli:
            p_val = float(kwargs.pop("p", 0.5))
            return dist.rvs(p_val, size=size, random_state=self._rng)
        if dist is hypergeom:
            M = int(kwargs.pop("M", 100))
            n = int(kwargs.pop("n", 30))
            N = int(kwargs.pop("N", 10))
            return dist.rvs(M, n, N, size=size, random_state=self._rng)
        if dist is chi2:
            df = float(kwargs.pop("df", 4))
            return dist.rvs(df, size=size, random_state=self._rng)
        if dist is f:
            dfn = float(kwargs.pop("dfn", 5))
            dfd = float(kwargs.pop("dfd", 2))
            return dist.rvs(dfn, dfd, size=size, random_state=self._rng)

        # Remaining distributions either take shape parameters or rely on ``fit``
        # to derive sensible defaults.
        if dist in {lognorm, gamma, beta, weibull_min}:
            args = kwargs.pop("args", (1.0,))
            return dist.rvs(*args, size=size, random_state=self._rng, **kwargs)
        return dist.rvs(size=size, random_state=self._rng, **kwargs)

    def distribution_diagnostics(self, samples: Iterable[float]) -> Dict[str, float]:
        """Return descriptive statistics for the supplied sample."""

        array = np.asarray(list(samples), dtype=float)
        if array.size == 0:
            raise HTTPException(status_code=400, detail="Cannot analyse an empty sample")
        description = stats.describe(array)
        return {
            "mean": float(description.mean),
            "variance": float(description.variance),
            "skewness": float(stats.skew(array)),
            "kurtosis": float(stats.kurtosis(array)),
            "min": float(description.minmax[0]),
            "max": float(description.minmax[1]),
        }

    # ------------------------------------------------------------------
    # Optimisation and scenario exploration
    # ------------------------------------------------------------------
    def optimise_parameter(self, parameter: str, target_metric: str, bounds: tuple[float, float]) -> Dict[str, Any]:
        """Optimise a global input parameter to approach a target metric."""

        table = self.model.input_page.global_inputs
        if table.placeholder or parameter not in table.data["Parameter"].values:
            raise HTTPException(status_code=404, detail=f"Parameter '{parameter}' not found")

        def objective(value: float) -> float:
            table.data.loc[table.data["Parameter"] == parameter, "Value"] = value
            metrics = self.model.build()["metrics"]
            return float(metrics.get(target_metric, np.nan))

        def _min_func(value: np.ndarray) -> float:
            metric = objective(float(value[0]))
            if not np.isfinite(metric):
                return np.finfo(float).max
            return -metric  # maximise the metric by minimising the negative value

        lower, upper = bounds
        result = minimize(_min_func, x0=np.array([(lower + upper) / 2.0]), bounds=[bounds])
        best_value = float(result.x[0])
        table.data.loc[table.data["Parameter"] == parameter, "Value"] = best_value
        metrics = self.model.build()["metrics"]
        return {
            "parameter": parameter,
            "value": best_value,
            "target_metric": target_metric,
            "metric_value": float(metrics.get(target_metric, np.nan)),
            "success": bool(result.success),
        }

    # ------------------------------------------------------------------
    # Visualisation utilities
    # ------------------------------------------------------------------
    def build_dependency_graph(self, dependencies: Mapping[str, Sequence[str]]) -> go.Figure:
        """Create an interactive dependency graph for the supplied mapping."""

        graph = nx.DiGraph()
        for node, edges in dependencies.items():
            graph.add_node(node)
            for edge in edges:
                graph.add_edge(node, edge)

        pos = nx.spring_layout(graph, seed=42)
        edge_x: List[float] = []
        edge_y: List[float] = []
        for source, target in graph.edges():
            edge_x.extend([pos[source][0], pos[target][0], None])
            edge_y.extend([pos[source][1], pos[target][1], None])

        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color="#888"), hoverinfo="none", mode="lines")
        node_x = [pos[node][0] for node in graph.nodes()]
        node_y = [pos[node][1] for node in graph.nodes()]
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=list(graph.nodes()),
            textposition="bottom center",
            marker=dict(size=12, color="#1f77b4", line=dict(width=1, color="#fff")),
        )
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
        return fig

    def render_plotly_html(self, figure: go.Figure) -> str:
        """Return a standalone HTML snippet for embedding Plotly figures."""

        return pio.to_html(figure, full_html=False, include_plotlyjs="cdn")

    def render_distribution_plot(self, samples: Sequence[float]) -> str:
        """Render a Matplotlib histogram and return the base64 encoded PNG."""

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(samples, bins=30, color="#4c72b0", alpha=0.8)
        ax.set_title("Distribution histogram")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        buffer = BytesIO()
        fig.tight_layout()
        fig.savefig(buffer, format="png")
        plt.close(fig)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("ascii")

    # ------------------------------------------------------------------
    # Miscellaneous helpers
    # ------------------------------------------------------------------
    def safe_add_worksheet(self, workbook, name: str):
        """Add a worksheet while providing a friendly HTTP error on duplicates."""

        try:
            return workbook.add_worksheet(name)
        except DuplicateWorksheetName as exc:
            raise HTTPException(status_code=400, detail=f"Worksheet '{name}' already exists") from exc

    @run_with_timeout(timeout_seconds=30)
    def heavy_simulation(self, iterations: int = 10_000) -> Dict[str, float]:
        """Run a CPU intensive Monte Carlo style simulation with a timeout."""

        samples = self.sample_distribution("normal", size=iterations)
        diagnostics = self.distribution_diagnostics(samples)
        diagnostics["iterations"] = float(iterations)
        diagnostics["scenario"] = getattr(self.model, "scenario", "unknown")
        return diagnostics

    def export_dependency_graph(self, dependencies: Mapping[str, Sequence[str]], output_path: os.PathLike[str] | str) -> str:
        """Persist a dependency graph figure to disk and return the path."""

        figure = self.build_dependency_graph(dependencies)
        html = self.render_plotly_html(figure)
        path = os.fspath(output_path)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(html)
        return path


# ``CassavaBioethanolModel`` is imported at the bottom to avoid a circular
# dependency with ``financial_model`` which instantiates :class:`AdvancedAnalyticsToolkit`.
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from .financial_model import CassavaBioethanolModel

