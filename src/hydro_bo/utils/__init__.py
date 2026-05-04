"""Cross-cutting utilities used by both the BO and MPC layers.

  - `logging_config`: structured logging setup (`configure_logging`, `get_logger`).
  - `seeding`: deterministic seed derivation (`resolve_master_seed`,
    `derive_subseed`, `make_rng`, etc.).
  - `jax_threads`: XLA / BLAS thread configuration. Must be called by
    scripts BEFORE the first `import jax`.
  - `run_config`: loader for the shared scripts/config.yml — typed
    sections for general / sobol / unconstrained_bo / constrained_bo.
  - `search_space`: PARAM_KEYS, INTEGER_KEYS, ABSOLUTE_BOUNDS, and
    `build_bounds` / `build_cat_vars` / `params_from_x` /
    `sobol_unit_sample` helpers shared by every run script.
"""

from hydro_bo.utils.logging_config import configure_logging, get_logger

__all__ = ["configure_logging", "get_logger"]
