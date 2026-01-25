"""Proxy to top-level data package for `hydro_bo.data` imports."""

import importlib

_data = importlib.import_module("data")
__all__ = getattr(_data, "__all__", []) or [
    name for name in dir(_data) if not name.startswith("_")
]

for _name in __all__:
    globals()[_name] = getattr(_data, _name)
