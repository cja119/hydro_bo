from hydro_bo.utils import configure_logging
from hydro_bo.mpc import MPCController
from hydro_bo.envs import ShippingEnv, Planning

# `MeanVarBayesopt` (and the back-compat `BayesianOptimizer` alias) are
# exposed lazily so that `import hydro_bo` does not eagerly import jax —
# scripts can call `configure_jax_threads` before triggering the import.
__all__ = [
    "BayesianOptimizer",
    "MeanVarBayesopt",
    "ConstrainedBayesopt",
    "MPCController",
    "ShippingEnv",
    "Planning",
    "configure_logging",
]


def __getattr__(name):
    if name in {"BayesianOptimizer", "MeanVarBayesopt", "ConstrainedBayesopt"}:
        from hydro_bo.opt import (
            BayesianOptimizer,
            MeanVarBayesopt,
            ConstrainedBayesopt,
        )

        return {
            "BayesianOptimizer": BayesianOptimizer,
            "MeanVarBayesopt": MeanVarBayesopt,
            "ConstrainedBayesopt": ConstrainedBayesopt,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
