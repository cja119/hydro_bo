"""Proxy to top-level algs namespace for `hydro_bo.algs` imports."""
import importlib

_algs = importlib.import_module("algs")
__all__ = getattr(_algs, "__all__", []) or [
    name for name in dir(_algs) if not name.startswith("_")
]

# Populate this module's globals to mirror the upstream package
for _name in __all__:
    globals()[_name] = getattr(_algs, _name)

