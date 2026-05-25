from __future__ import annotations

import math
from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

try:  # pragma: no cover - import fast path
    from scipy import stats as _stats  # type: ignore
    from scipy.optimize import minimize as _minimize  # type: ignore
except Exception:  # pragma: no cover - SciPy unavailable
    _stats = None
    _minimize = None


def _as_rng(random_state: Any | None) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    if isinstance(random_state, np.random.RandomState):  # pragma: no cover - legacy support
        seed = random_state.randint(0, 2**32 - 1)
        return np.random.default_rng(int(seed))
    if random_state is None:
        return np.random.default_rng()
    return np.random.default_rng(int(random_state))


class _RVDistribution:
    def __init__(self, sampler):
        self._sampler = sampler

    def rvs(self, *shape_args, random_state=None, size=None, **kwargs):
        rng = _as_rng(random_state)
        return self._sampler(rng, shape_args, size=size, **kwargs)


def _normal_sampler(rng, shape_args, *, size=None, loc=0.0, scale=1.0, **_):
    return rng.normal(loc=loc, scale=scale, size=size)


def _lognormal_sampler(rng, shape_args, *, size=None, loc=0.0, scale=1.0, **_):
    sigma = float(shape_args[0] if shape_args else 1.0)
    if scale <= 0:
        scale = 1.0
    samples = rng.lognormal(mean=math.log(scale), sigma=sigma, size=size)
    return loc + samples


def _uniform_sampler(rng, shape_args, *, size=None, loc=0.0, scale=1.0, **_):
    return rng.uniform(low=loc, high=loc + scale, size=size)


def _exponential_sampler(rng, shape_args, *, size=None, loc=0.0, scale=1.0, **_):
    return loc + rng.exponential(scale=scale, size=size)


def _binomial_sampler(rng, shape_args, *, size=None, loc=0.0, **kwargs):
    if shape_args:
        n, p = shape_args
    else:
        n = kwargs.get("n", 1)
        p = kwargs.get("p", 0.5)
    draws = rng.binomial(int(n), float(p), size=size)
    return loc + draws


def _poisson_sampler(rng, shape_args, *, size=None, loc=0.0, **kwargs):
    mu = shape_args[0] if shape_args else kwargs.get("mu", 1.0)
    draws = rng.poisson(float(mu), size=size)
    return loc + draws


def _geometric_sampler(rng, shape_args, *, size=None, loc=0.0, **kwargs):
    p = shape_args[0] if shape_args else kwargs.get("p", 0.5)
    draws = rng.geometric(float(p), size=size)
    return loc + draws


def _bernoulli_sampler(rng, shape_args, *, size=None, loc=0.0, **kwargs):
    p = shape_args[0] if shape_args else kwargs.get("p", 0.5)
    draws = rng.binomial(1, float(p), size=size)
    return loc + draws


def _chi2_sampler(rng, shape_args, *, size=None, loc=0.0, scale=1.0, **kwargs):
    df = shape_args[0] if shape_args else kwargs.get("df", 1.0)
    draws = rng.chisquare(float(df), size=size)
    return loc + scale * draws


def _gamma_sampler(rng, shape_args, *, size=None, loc=0.0, scale=1.0, **kwargs):
    a = shape_args[0] if shape_args else kwargs.get("a", 1.0)
    draws = rng.gamma(float(a), scale=scale, size=size)
    return loc + draws


def _weibull_sampler(rng, shape_args, *, size=None, loc=0.0, scale=1.0, **kwargs):
    c = shape_args[0] if shape_args else kwargs.get("c", 1.0)
    draws = rng.weibull(float(c), size=size)
    return loc + scale * draws


def _hypergeom_sampler(rng, shape_args, *, size=None, loc=0.0, **kwargs):
    if shape_args:
        M, n, N = shape_args
    else:
        M = kwargs.get("M", 100)
        n = kwargs.get("n", 30)
        N = kwargs.get("N", 10)
    draws = rng.hypergeometric(int(n), int(M) - int(n), int(N), size=size)
    return loc + draws


def _multinomial_sampler(rng, shape_args, *, size=None, **kwargs):
    n = shape_args[0] if shape_args else kwargs.get("n", 1)
    pvals = kwargs.get("pvals")
    if pvals is None and len(shape_args) > 1:
        pvals = shape_args[1]
    if isinstance(pvals, str):
        pvals = [float(part.strip()) for part in pvals.split(",") if part.strip()]
    if pvals is None:
        pvals = np.full(2, 0.5, dtype=float)
    pvals = np.asarray(pvals, dtype=float)
    total = pvals.sum()
    if total <= 0:
        pvals = np.full_like(pvals, 1 / pvals.size)
    else:
        pvals = pvals / total
    draws = rng.multinomial(int(n), pvals, size=size)
    return draws


def _beta_sampler(rng, shape_args, *, size=None, loc=0.0, scale=1.0, **kwargs):
    a = shape_args[0] if shape_args else kwargs.get("a", 1.0)
    b = shape_args[1] if len(shape_args) > 1 else kwargs.get("b", 1.0)
    draws = rng.beta(float(a), float(b), size=size)
    return loc + scale * draws


def _f_sampler(rng, shape_args, *, size=None, loc=0.0, scale=1.0, **kwargs):
    dfn = shape_args[0] if shape_args else kwargs.get("dfn", 1.0)
    dfd = shape_args[1] if len(shape_args) > 1 else kwargs.get("dfd", 1.0)
    draws = rng.f(float(dfn), float(dfd), size=size)
    return loc + scale * draws


if _stats is not None and _minimize is not None:
    stats = _stats
    minimize = _minimize
    bernoulli = stats.bernoulli
    beta = stats.beta
    binom = stats.binom
    chi2 = stats.chi2
    expon = stats.expon
    f = stats.f
    gamma = stats.gamma
    geom = stats.geom
    hypergeom = stats.hypergeom
    lognorm = stats.lognorm
    multinomial = stats.multinomial
    norm = stats.norm
    poisson = stats.poisson
    uniform = stats.uniform
    weibull_min = stats.weibull_min
else:  # pragma: no cover - exercised via tests when SciPy missing
    class _StatsModule:
        def __init__(self) -> None:
            self.norm = _RVDistribution(_normal_sampler)
            self.lognorm = _RVDistribution(_lognormal_sampler)
            self.uniform = _RVDistribution(_uniform_sampler)
            self.expon = _RVDistribution(_exponential_sampler)
            self.binom = _RVDistribution(_binomial_sampler)
            self.poisson = _RVDistribution(_poisson_sampler)
            self.geom = _RVDistribution(_geometric_sampler)
            self.bernoulli = _RVDistribution(_bernoulli_sampler)
            self.chi2 = _RVDistribution(_chi2_sampler)
            self.gamma = _RVDistribution(_gamma_sampler)
            self.weibull_min = _RVDistribution(_weibull_sampler)
            self.hypergeom = _RVDistribution(_hypergeom_sampler)
            self.multinomial = _RVDistribution(_multinomial_sampler)
            self.beta = _RVDistribution(_beta_sampler)
            self.f = _RVDistribution(_f_sampler)

        def describe(self, data: Iterable[float]):
            array = np.asarray(list(data), dtype=float)
            if array.size == 0:
                raise ValueError("describe requires at least one data point")
            n = int(array.size)
            mean = float(array.mean())
            if n > 1:
                variance = float(array.var(ddof=1))
            else:
                variance = 0.0
            minmax = (float(array.min()), float(array.max()))
            skewness = float(self.skew(array))
            kurtosis = float(self.kurtosis(array))
            DescribeResult = namedtuple(
                "DescribeResult", ["nobs", "minmax", "mean", "variance", "skewness", "kurtosis"]
            )
            return DescribeResult(n, minmax, mean, variance, skewness, kurtosis)

        def skew(self, data: Iterable[float]) -> float:
            array = np.asarray(list(data), dtype=float)
            if array.size < 2:
                return 0.0
            mean = array.mean()
            centered = array - mean
            std = centered.std(ddof=0)
            if std == 0:
                return 0.0
            m3 = np.mean(centered**3)
            return float(m3 / std**3)

        def kurtosis(self, data: Iterable[float]) -> float:
            array = np.asarray(list(data), dtype=float)
            if array.size < 2:
                return -3.0
            mean = array.mean()
            centered = array - mean
            std = centered.std(ddof=0)
            if std == 0:
                return -3.0
            m4 = np.mean(centered**4)
            return float(m4 / std**4 - 3.0)

    stats = _StatsModule()
    bernoulli = stats.bernoulli
    beta = stats.beta
    binom = stats.binom
    chi2 = stats.chi2
    expon = stats.expon
    f = stats.f
    gamma = stats.gamma
    geom = stats.geom
    hypergeom = stats.hypergeom
    lognorm = stats.lognorm
    multinomial = stats.multinomial
    norm = stats.norm
    poisson = stats.poisson
    uniform = stats.uniform
    weibull_min = stats.weibull_min

    @dataclass
    class _MinimizeResult:
        x: np.ndarray
        success: bool
        fun: float
        nfev: int
        nit: int
        message: str

    def minimize(func, x0: Sequence[float], bounds: Sequence[tuple[float, float]] | None = None, tol: float = 1e-6, maxiter: int = 200):
        if bounds:
            lower, upper = bounds[0]
        else:
            lower, upper = x0[0] - 1.0, x0[0] + 1.0
        phi = (1 + 5 ** 0.5) / 2
        inv_phi = 1 / phi
        inv_phi_sq = inv_phi**2
        a, b = float(lower), float(upper)
        c = b - inv_phi * (b - a)
        d = a + inv_phi * (b - a)
        fc = func(np.array([c]))
        fd = func(np.array([d]))
        nfev = 2
        nit = 0
        while abs(b - a) > tol and nit < maxiter:
            nit += 1
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - inv_phi * (b - a)
                fc = func(np.array([c]))
                nfev += 1
            else:
                a, c, fc = c, d, fd
                d = a + inv_phi * (b - a)
                fd = func(np.array([d]))
                nfev += 1
        x = np.array([(a + b) / 2.0])
        fun = func(x)
        nfev += 1
        return _MinimizeResult(x=x, success=True, fun=float(fun), nfev=nfev, nit=nit, message="success")

__all__ = [
    "stats",
    "minimize",
    "bernoulli",
    "beta",
    "binom",
    "chi2",
    "expon",
    "f",
    "gamma",
    "geom",
    "hypergeom",
    "lognorm",
    "multinomial",
    "norm",
    "poisson",
    "uniform",
    "weibull_min",
]
