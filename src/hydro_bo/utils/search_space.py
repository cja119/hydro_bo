"""Shared search-space layout for the BO and Sobol scripts.

`PARAM_KEYS` is the ordered list of decision variables the BO searches
over; `INTEGER_KEYS` marks those whose feasible values are integers (the
BO branches over them); `ABSOLUTE_BOUNDS` overrides the ±expansion
bounds for entries with absolute ranges (e.g. backoff fractions).

`sobol_unit_sample` regenerates a deterministic Sobol row from a
`(seed, pow_n)` pair — replaces the old sobol_indices.csv: every script
run with the same config sees the same hypercube point at the same row.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats.qmc import Sobol


PARAM_KEYS = [
    "compression_capacity",
    "conversion_trains_number",
    "electrolyser_capacity",
    "fuelcell_capacity",
    "hydrogen_storage_capacity",
    "renewable_energy_capacity",
    "vector_storage_capacity",
    "hydrogen_storage_lower_backoff",
    "hydrogen_storage_upper_backoff",
    "vector_storage_lower_backoff",
    "vector_storage_upper_backoff",
    "wind_forecast_mean",
    "expected_arrival_offset",
]

# Dims the BO branches over as integer levels.
# `expected_arrival_offset` lives in days, so its absolute bounds (-2, 2)
# resolve to five categorical levels {-2,-1,0,1,2} via build_cat_vars.
INTEGER_KEYS = {"conversion_trains_number", "expected_arrival_offset"}

ABSOLUTE_BOUNDS = {
    "hydrogen_storage_lower_backoff": (0.0, 0.5),
    "hydrogen_storage_upper_backoff": (0.5, 1.0),
    "vector_storage_lower_backoff":   (0.0, 0.5),
    "vector_storage_upper_backoff":   (0.5, 1.0),
    "wind_forecast_mean":             (0.0, 1.0),
    "expected_arrival_offset":        (-2.0, 2.0),
}

# Search dims that are env-side overrides (not planning-model params).

ENV_OVERRIDE_PATHS = {
    "wind_forecast_mean": ("weather_data", "forecast_mean_override"),
}
ENV_OVERRIDE_KEYS = set(ENV_OVERRIDE_PATHS)


def _set_nested(target: dict, path: tuple, value) -> None:
    """Set `value` at `path` (a tuple of keys) inside nested dict `target`,
    creating intermediate dicts as needed."""
    d = target
    for key in path[:-1]:
        d = d.setdefault(key, {})
    d[path[-1]] = value


def split_env_overrides(params: dict) -> tuple[dict, dict]:
    """Split a flat params dict into `(planning_params, env_overrides)`.

    Any key in `ENV_OVERRIDE_PATHS` is popped out of the planning params
    and placed at its nested path in `env_overrides`, ready to deep-merge
    into `env_args["config"]`. Mirrors what `params_from_x` does for a BO
    sample vector, but operates on an already-decoded param dict (e.g. an
    injected planning-model YAML for the multisim)."""
    planning = dict(params)
    env_overrides: dict = {}
    for key, path in ENV_OVERRIDE_PATHS.items():
        if key in planning:
            _set_nested(env_overrides, path, float(planning.pop(key)))
    return planning, env_overrides


def build_bounds(ref: dict, expansion: float) -> np.ndarray:
    """`ABSOLUTE_BOUNDS` keys use their fixed ranges; everything else
    gets ±expansion around the planning-model reference value."""
    bounds = []
    for key in PARAM_KEYS:
        if key in ABSOLUTE_BOUNDS:
            bounds.append(list(ABSOLUTE_BOUNDS[key]))
            continue
        val = float(ref[key])
        lo = val * (1.0 - expansion)
        hi = val * (1.0 + expansion)
        if key == "conversion_trains_number":
            lo = max(1.0, lo)
        bounds.append([lo, hi])
    return np.array(bounds)


def build_cat_vars(bounds: np.ndarray) -> list[tuple[int, list[float]]]:
    """Enumerate integer levels of each integer dim and project them to
    unit-cube positions for the BO branch optimiser."""
    cat_vars: list[tuple[int, list[float]]] = []
    for i, key in enumerate(PARAM_KEYS):
        if key not in INTEGER_KEYS:
            continue
        lo, hi = float(bounds[i, 0]), float(bounds[i, 1])
        span = hi - lo
        if span <= 0:
            raise ValueError(f"Integer key {key!r} has degenerate bounds [{lo}, {hi}]")
        lo_int = int(math.ceil(lo))
        hi_int = int(math.floor(hi))
        if hi_int < lo_int:
            raise ValueError(f"Integer key {key!r} bounds [{lo}, {hi}] contain no integers")
        unit_positions = [(k - lo) / span for k in range(lo_int, hi_int + 1)]
        cat_vars.append((i, unit_positions))
    return cat_vars


def params_from_x(
    x: np.ndarray, ref: dict, *, renewables: str, vector: str,
    cost_overrides: dict | None = None,
) -> tuple[dict, dict]:
    """Convert a BO sample vector into `(planning_params, env_overrides)`.

    `planning_params` flows into `RayMultiMPC(param_overrides=...)` —
    capacities, backoffs, integer counts, the MPC-side
    `expected_arrival_offset`, plus computed `capex`/`opex`.

    `env_overrides` flows into `env_args["config"]` via deep-merge —
    env-side knobs not represented in the planning-model reference dict
    (currently just the wind forecast mean, nested under `weather_data`).
    """
    from hydro_bo.envs.shipping.utils import calculate_capex_opex

    p = dict(ref)
    env_overrides: dict = {}
    for i, key in enumerate(PARAM_KEYS):
        val = float(x[i])
        if key in INTEGER_KEYS:
            val = int(round(val))
            if key == "conversion_trains_number":
                val = max(1, val)
        if key in ENV_OVERRIDE_PATHS:
            _set_nested(env_overrides, ENV_OVERRIDE_PATHS[key], float(val))
            continue
        p[key] = val

    costs = calculate_capex_opex(
        renewables=renewables,
        vector=vector,
        compression_capacity=p["compression_capacity"],
        electrolyser_capacity=p["electrolyser_capacity"],
        fuelcell_capacity=p["fuelcell_capacity"],
        conversion_trains_number=int(p["conversion_trains_number"]),
        hydrogen_storage_capacity=p["hydrogen_storage_capacity"],
        renewable_energy_capacity=p["renewable_energy_capacity"],
        vector_storage_capacity=p["vector_storage_capacity"],
        cost_overrides=cost_overrides,
    )
    p["capex"] = costs["capex"]
    p["opex"] = costs["opex"]
    return p, env_overrides


def flatten_dims(planning_params: dict, env_overrides: dict) -> dict:
    """Flat dict keyed by `PARAM_KEYS`, sourcing values from
    `planning_params` for planning-side dims and from `env_overrides`
    for env-side dims. Used by the BO/Sobol scripts so the per-eval log
    rows have one column per BO dim regardless of where the value lives.
    """
    out: dict = {}
    for key in PARAM_KEYS:
        if key in ENV_OVERRIDE_PATHS:
            d = env_overrides
            for part in ENV_OVERRIDE_PATHS[key]:
                d = d[part]
            out[key] = d
        else:
            out[key] = planning_params[key]
    return out


def sobol_unit_sample(
    seed: int, pow_n: int, index_row: int, dim: int | None = None
) -> np.ndarray:
    """Regenerate the Sobol sequence with `seed` and return row
    `index_row`. Replaces sobol_indices.csv — every script run with the
    same `(seed, pow_n)` sees the same hypercube point at the same
    index, so PBS array tasks (and downstream BO loaders) stay in sync
    without a shared CSV.

    `dim` defaults to the design space; pass the joint dimension
    (design + theta) for the parametric runs."""
    n = 2**pow_n
    if index_row < 0 or index_row >= n:
        raise IndexError(f"index_row={index_row} out of range [0, {n})")
    sampler = Sobol(d=dim if dim is not None else len(PARAM_KEYS),
                    scramble=True, seed=seed)
    return sampler.random(n)[index_row]


def scale_unit_to_bounds(unit: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    lo, hi = bounds[:, 0], bounds[:, 1]
    return lo + unit * (hi - lo)
