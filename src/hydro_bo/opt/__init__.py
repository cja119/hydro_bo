"""Bayesian optimisation package.

Importing this package does NOT eagerly import jax — that only happens
when you touch one of the surrogate / acquisition / BO modules. Scripts
should call `hydro_bo.utils.jax_threads.configure_jax_threads`
BEFORE importing any name from this package other than `jax_threads`
itself, so XLA picks up the right thread / X64 settings at first
`import jax`.
"""

__all__ = [
    "BaseBayesopt",
    "MeanVarBayesopt",
    "ConstrainedBayesopt",
    "BayesianOptimizer",
]


def __getattr__(name):
    if name in __all__:
        from hydro_bo.opt.bayesopt import (
            BaseBayesopt,
            MeanVarBayesopt,
            ConstrainedBayesopt,
            BayesianOptimizer,
        )

        return {
            "BaseBayesopt": BaseBayesopt,
            "MeanVarBayesopt": MeanVarBayesopt,
            "ConstrainedBayesopt": ConstrainedBayesopt,
            "BayesianOptimizer": BayesianOptimizer,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
