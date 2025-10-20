"""Fallback NumPy shim used when the compiled dependency is unavailable."""

from __future__ import annotations

import sys

from tools.numpy_stub import install_numpy_stub as _install_numpy_stub

_module = _install_numpy_stub(target=sys.modules[__name__])

globals().update({name: getattr(_module, name) for name in dir(_module) if not name.startswith("_")})
__all__ = [name for name in dir(_module) if not name.startswith("_")]
