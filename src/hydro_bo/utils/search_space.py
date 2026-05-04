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
]

INTEGER_KEYS = {"conversion_trains_number"}

ABSOLUTE_BOUNDS = {
    "hydrogen_storage_lower_backoff": (0.0, 0.5),
    "hydrogen_storage_upper_backoff": (0.5, 1.0),
    "vector_storage_lower_backoff":   (0.0, 0.5),
    "vector_storage_upper_backoff":   (0.5, 1.0),
}


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


def params_from_x(x: np.ndarray, ref: dict, *, renewables: str, vector: str) -> dict:
    """Convert a BO sample vector into a full planning model dict."""
    from hydro_bo.envs.shipping.utils import calculate_capex_opex

    p = dict(ref)
    for i, key in enumerate(PARAM_KEYS):
        val = float(x[i])
        if key == "conversion_trains_number":
            val = max(1, int(round(val)))
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
    )
    p["capex"] = costs["capex"]
    p["opex"] = costs["opex"]
    return p


def sobol_unit_sample(seed: int, pow_n: int, index_row: int) -> np.ndarray:
    """Regenerate the Sobol sequence with `seed` and return row
    `index_row`. Replaces sobol_indices.csv — every script run with the
    same `(seed, pow_n)` sees the same hypercube point at the same
    index, so PBS array tasks (and downstream BO loaders) stay in sync
    without a shared CSV."""
    n = 2**pow_n
    if index_row < 0 or index_row >= n:
        raise IndexError(f"index_row={index_row} out of range [0, {n})")
    sampler = Sobol(d=len(PARAM_KEYS), scramble=True, seed=seed)
    return sampler.random(n)[index_row]


def scale_unit_to_bounds(unit: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    lo, hi = bounds[:, 0], bounds[:, 1]
    return lo + unit * (hi - lo)
