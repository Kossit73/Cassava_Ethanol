"""Lightweight numpy compatibility layer for offline environments.

This module installs a very small subset of NumPy's API so that the
application can continue to run when the compiled dependency is unavailable.
Only the functions and classes required by the fallback modelling pipeline
are implemented.  The intent is not to provide numerical parity with NumPy
but simply to offer convenient helpers (``np.array``, ``np.isfinite`` and
friends) that behave well enough for the pure-Python model.
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Callable, Iterable, Iterator, List, Sequence, Tuple, TypeVar, Union

T = TypeVar("T")


def _as_iterable(values: Union["ndarray", Sequence[T], T]) -> List[T]:
    if isinstance(values, ndarray):
        return list(values)
    if isinstance(values, (list, tuple)):
        return list(values)  # type: ignore[return-value]
    if isinstance(values, (str, bytes)):
        return [values]  # type: ignore[list-item]
    if isinstance(values, Iterable):
        return list(values)  # type: ignore[return-value]
    return [values]  # type: ignore[list-item]


class ndarray(List[float]):
    """Tiny stand-in for :class:`numpy.ndarray` with element-wise ops."""

    def __init__(self, values: Iterable[float] = ()) -> None:  # pragma: no cover - simple container
        super().__init__(values)

    # Vectorised helpers -------------------------------------------------
    def _binary_op(self, other: Union[float, Sequence[float]], op: Callable[[float, float], float]):
        if isinstance(other, (int, float)):
            return ndarray(op(a, float(other)) for a in self)
        other_seq = _as_iterable(other)
        length = min(len(self), len(other_seq))
        return ndarray(op(self[i], float(other_seq[i])) for i in range(length))

    def __add__(self, other):
        return self._binary_op(other, lambda a, b: a + b)

    def __radd__(self, other):  # pragma: no cover - symmetric with __add__
        return self.__add__(other)

    def __sub__(self, other):
        return self._binary_op(other, lambda a, b: a - b)

    def __rsub__(self, other):  # pragma: no cover - symmetric with __sub__
        if isinstance(other, (int, float)):
            return ndarray(float(other) - a for a in self)
        other_seq = _as_iterable(other)
        length = min(len(self), len(other_seq))
        return ndarray(float(other_seq[i]) - self[i] for i in range(length))

    def __mul__(self, other):
        return self._binary_op(other, lambda a, b: a * b)

    def __rmul__(self, other):  # pragma: no cover - symmetric with __mul__
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._binary_op(other, lambda a, b: a / b if b != 0 else float("inf"))

    def __rtruediv__(self, other):  # pragma: no cover - symmetric with __truediv__
        if isinstance(other, (int, float)):
            return ndarray(float(other) / a if a != 0 else float("inf") for a in self)
        other_seq = _as_iterable(other)
        length = min(len(self), len(other_seq))
        return ndarray(float(other_seq[i]) / self[i] if self[i] != 0 else float("inf") for i in range(length))

    def __pow__(self, power):  # pragma: no cover - rarely used
        if isinstance(power, (int, float)):
            return ndarray(a ** float(power) for a in self)
        power_seq = _as_iterable(power)
        length = min(len(self), len(power_seq))
        return ndarray(self[i] ** float(power_seq[i]) for i in range(length))

    def __eq__(self, other):  # pragma: no cover - trivial comparison
        if isinstance(other, (int, float)):
            return ndarray(1 if a == float(other) else 0 for a in self)
        other_seq = _as_iterable(other)
        length = min(len(self), len(other_seq))
        return ndarray(1 if self[i] == float(other_seq[i]) else 0 for i in range(length))

    def __ne__(self, other):  # pragma: no cover - trivial comparison
        result = self.__eq__(other)
        return ndarray(1 - value for value in result)

    def __neg__(self):  # pragma: no cover - rarely used
        return ndarray(-a for a in self)

    @property
    def shape(self) -> Tuple[int, ...]:  # pragma: no cover - trivial
        return (len(self),)

    def tolist(self) -> List[float]:  # pragma: no cover - convenience helper
        return list(self)


def _elementwise(value, func: Callable[[float], float]):
    if isinstance(value, ndarray):
        return ndarray(func(v) for v in value)
    if isinstance(value, (list, tuple)):
        return ndarray(func(float(v)) for v in value)
    return func(float(value))


def array(values: Union[ndarray, Sequence[float], float]) -> ndarray:
    return ndarray(_as_iterable(values))


def asarray(values: Union[ndarray, Sequence[float], float]) -> ndarray:  # pragma: no cover - alias
    return array(values)


def arange(start, stop=None, step=1):
    if stop is None:
        start, stop = 0, start
    if step == 0:
        raise ValueError("step must be non-zero")
    values = []
    current = float(start)
    comparison = (lambda a, b: a < b) if step > 0 else (lambda a, b: a > b)
    while comparison(current, float(stop)):
        values.append(current)
        current += step
    return ndarray(values)


def argsort(values):  # pragma: no cover - deterministic helper
    seq = _as_iterable(values)
    return ndarray(sorted(range(len(seq)), key=seq.__getitem__))


def flatnonzero(values):  # pragma: no cover - deterministic helper
    seq = _as_iterable(values)
    return ndarray(idx for idx, value in enumerate(seq) if float(value) != 0.0)


def isfinite(values):
    return _elementwise(values, lambda x: bool(math.isfinite(x)))


def isnan(values):  # pragma: no cover - deterministic helper
    return _elementwise(values, lambda x: bool(math.isnan(x)))


def isclose(a, b, rtol=1e-05, atol=1e-08):  # pragma: no cover - deterministic helper
    def _close(x, y):
        return bool(abs(x - y) <= (atol + rtol * abs(y)))

    if isinstance(a, (ndarray, list, tuple)) or isinstance(b, (ndarray, list, tuple)):
        seq_a = _as_iterable(a)
        seq_b = _as_iterable(b)
        length = min(len(seq_a), len(seq_b))
        return ndarray(_close(float(seq_a[i]), float(seq_b[i])) for i in range(length))
    return bool(_close(float(a), float(b)))


def clip(values, a_min, a_max):  # pragma: no cover - deterministic helper
    def _clip(value: float) -> float:
        if a_min is not None:
            value = max(value, float(a_min))
        if a_max is not None:
            value = min(value, float(a_max))
        return value

    return _elementwise(values, _clip)


def where(condition, x, y):  # pragma: no cover - deterministic helper
    cond_seq = _as_iterable(condition)
    x_seq = _as_iterable(x)
    y_seq = _as_iterable(y)
    length = min(len(cond_seq), len(x_seq), len(y_seq))
    return ndarray(x_seq[i] if cond_seq[i] else y_seq[i] for i in range(length))


def all(values):  # pragma: no cover - trivial helper
    return builtins_all(bool(v) for v in _as_iterable(values))


def any(values):  # pragma: no cover - trivial helper
    return builtins_any(bool(v) for v in _as_iterable(values))


def cumsum(values):  # pragma: no cover - deterministic helper
    total = 0.0
    cumulative = []
    for value in _as_iterable(values):
        total += float(value)
        cumulative.append(total)
    return ndarray(cumulative)


def argmax(values):  # pragma: no cover - deterministic helper
    seq = _as_iterable(values)
    if not seq:
        return 0
    max_index = 0
    max_value = seq[0]
    for idx, value in enumerate(seq):
        if value > max_value:
            max_index = idx
            max_value = value
    return max_index


def zeros(length):  # pragma: no cover - deterministic helper
    return ndarray(0.0 for _ in range(int(length)))


def ones(length):  # pragma: no cover - deterministic helper
    return ndarray(1.0 for _ in range(int(length)))


def full(length, fill_value):  # pragma: no cover - deterministic helper
    return ndarray(float(fill_value) for _ in range(int(length)))


def finfo(dtype=float):  # pragma: no cover - deterministic helper
    @dataclass
    class _Finfo:
        max: float = sys.float_info.max
        min: float = -sys.float_info.max
        tiny: float = sys.float_info.min

    return _Finfo()


def copysign(x, y):  # pragma: no cover - deterministic helper
    return math.copysign(x, y)


def issubdtype(dtype, klass):  # pragma: no cover - heuristic helper
    try:
        dtype_type = dtype.type if hasattr(dtype, "type") else dtype
        if isinstance(klass, tuple):
            return any(issubdtype(dtype_type, item) for item in klass)
        if klass in (float, int):
            return issubclass(dtype_type, klass)  # type: ignore[arg-type]
        if klass is datetime64:
            name = getattr(dtype, "name", "")
            return "datetime" in str(name).lower()
    except Exception:
        return False
    return False


class _RandomGenerator:  # pragma: no cover - deterministic helper
    def __init__(self, seed=None) -> None:
        self._rng = random.Random(seed)

    def normal(self, mean=0.0, std=1.0, size=None):
        if size is None:
            return self._rng.gauss(mean, std)
        return ndarray(self._rng.gauss(mean, std) for _ in range(int(size)))

    def uniform(self, low=0.0, high=1.0, size=None):
        if size is None:
            return self._rng.uniform(low, high)
        return ndarray(self._rng.uniform(low, high) for _ in range(int(size)))

    def integers(self, low, high=None, size=None):
        if high is None:
            low, high = 0, low
        if size is None:
            return self._rng.randrange(low, high)
        return ndarray(self._rng.randrange(low, high) for _ in range(int(size)))


def default_rng(seed=None):  # pragma: no cover - deterministic helper
    return _RandomGenerator(seed)


def _populate_module(module: ModuleType) -> ModuleType:
    module.ndarray = ndarray
    module.number = (int, float)
    module.floating = float
    module.integer = int
    module.datetime64 = datetime64
    module.nan = float("nan")
    module.inf = float("inf")
    module.pi = math.pi

    module.array = array
    module.asarray = asarray
    module.arange = arange
    module.argsort = argsort
    module.flatnonzero = flatnonzero
    module.isfinite = isfinite
    module.isclose = isclose
    module.isnan = isnan
    module.clip = clip
    module.where = where
    module.all = all
    module.any = any
    module.cumsum = cumsum
    module.argmax = argmax
    module.zeros = zeros
    module.ones = ones
    module.full = full
    module.finfo = finfo
    module.copysign = copysign
    module.issubdtype = issubdtype
    return module


def _ensure_random_namespace(module: ModuleType) -> None:
    random_module = sys.modules.get("numpy.random")
    if random_module is None or not hasattr(random_module, "default_rng"):
        random_module = ModuleType("numpy.random")
        random_module.default_rng = default_rng
        sys.modules["numpy.random"] = random_module
    module.random = random_module


def install_numpy_stub(target: ModuleType | None = None) -> ModuleType:
    """Install the stub in :mod:`sys.modules` if NumPy is absent."""

    if target is None and "numpy" in sys.modules:
        existing = sys.modules["numpy"]
        if isinstance(existing, ModuleType) and hasattr(existing, "array"):
            return existing

    module = target if target is not None else ModuleType("numpy")
    _populate_module(module)
    sys.modules["numpy"] = module
    _ensure_random_namespace(module)
    return module


# ---------------------------------------------------------------------------
# Compatibility with the rest of the project
# ---------------------------------------------------------------------------

from builtins import all as builtins_all  # noqa: E402  (import after helper)
from builtins import any as builtins_any  # noqa: E402
from datetime import datetime as datetime64  # noqa: E402  - light alias

