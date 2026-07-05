from hydro_bo.utils import configure_logging

__all__ = [
    "MeanVarBayesopt",
    "ConstrainedBayesopt",
    "MPCController",
    "ShippingEnv",
    "Planning",
    "configure_logging",
]

# name -> (module, attribute) resolved on first access.
_LAZY = {
    "MeanVarBayesopt": ("hydro_bo.opt", "MeanVarBayesopt"),
    "ConstrainedBayesopt": ("hydro_bo.opt", "ConstrainedBayesopt"),
    "MPCController": ("hydro_bo.mpc", "MPCController"),
    "ShippingEnv": ("hydro_bo.envs", "ShippingEnv"),
    "Planning": ("hydro_bo.envs", "Planning"),
}


def __getattr__(name):
    if name in _LAZY:
        import importlib

        module_name, attr = _LAZY[name]
        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
