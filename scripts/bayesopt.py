"""
Bayesian optimisation over the MPC planning model parameters.

Objective: maximise   mean(scores) - penalty * var(scores)
           where scores are reward/tonne values from RayMultiMPC.run_multisim()

Search space: ±BOUNDS_EXPANSION % around a reference planning model file.
              capex/opex are recomputed from calculate_capex_opex at each sample.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
import numpy as np

from hydro_bo import configure_logging
from src.algs.dispatcher import RayMultiMPC
from src.algs.bayesopt import BayesianOptimizer
from src.envs.shipping.utils import calculate_capex_opex

configure_logging()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PLANNING_MODEL   = "NH3-Chile.yml"
VECTOR           = "NH3"
RENEWABLES       = "wind"
WEATHER_FILE     = "CoastalChile_15-20_Wind.csv"
REFERENCE_PATH   = Path(__file__).parent.parent / "src/tmp/planning" / PLANNING_MODEL

BOUNDS_EXPANSION = 0.5        # ±50 % around reference values
VARIANCE_PENALTY = 0.5         # λ in: mean - λ * var
N_INITIAL_POINTS = 8           # Sobol evaluations before BO starts
BO_ITER_LIMIT    = 20          # BO iterations
N_INSTANCES      = 8           # total MPC runs per BO evaluation (parallelised across NUM_DEVICES)
NUM_DEVICES      = 4           # simultaneous instances
TIMEOUT          = 900         # seconds per evaluation

ENV_ARGS = {
    "config": {
        "vector": VECTOR,
        "mpc": {"planning_model": PLANNING_MODEL},
        "weather_data": {"weather_file": WEATHER_FILE},
    },
}

# ---------------------------------------------------------------------------
# Parameters that form the search space (all continuous capacities).
# conversion_trains_number is integer — we round at sample time.
# renewables / vector / capex / opex are NOT search variables.
# ---------------------------------------------------------------------------

PARAM_KEYS = [
    "compression_capacity",
    "conversion_trains_number",   # integer — rounded at evaluation
    "electrolyser_capacity",
    "fuelcell_capacity",
    "hydrogen_storage_capacity",
    "renewable_energy_capacity",
    "vector_storage_capacity",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_reference(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_bounds(ref: dict, expansion: float) -> np.ndarray:
    """Return (len(PARAM_KEYS), 2) bounds array."""
    bounds = []
    for key in PARAM_KEYS:
        val = float(ref[key])
        lo = val * (1.0 - expansion)
        hi = val * (1.0 + expansion)
        # conversion_trains_number must be ≥ 1
        if key == "conversion_trains_number":
            lo = max(1.0, lo)
        bounds.append([lo, hi])
    return np.array(bounds)


def params_from_x(x: np.ndarray, ref: dict) -> dict:
    """Convert a BO sample vector into a full planning model dict."""
    p = dict(ref)   # carry over renewables, vector, etc.

    for i, key in enumerate(PARAM_KEYS):
        val = float(x[i])
        if key == "conversion_trains_number":
            val = max(1, int(round(val)))
        p[key] = val

    # Recompute derived capex / opex
    costs = calculate_capex_opex(
        renewables=RENEWABLES,
        vector=VECTOR,
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


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def objective(x: np.ndarray, ref: dict) -> float:
    """Evaluate mean - λ·var over N_INSTANCES MPC runs for parameter vector x."""
    params = params_from_x(x, ref)

    dispatcher = RayMultiMPC(
        env_args=ENV_ARGS,
        param_overrides=params,
        num_instances=N_INSTANCES,
        num_devices=NUM_DEVICES,
        timeout=TIMEOUT,
        exit_fraction=1.0,
    )

    scores = dispatcher.run_multisim()

    if not scores:
        return -np.inf

    scores_arr = np.array(scores, dtype=float)
    return float(scores_arr.mean() - VARIANCE_PENALTY * scores_arr.var())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_bayesopt():
    ref = load_reference(REFERENCE_PATH)
    bounds = build_bounds(ref, BOUNDS_EXPANSION)

    print(f"[BO] Search space ({len(PARAM_KEYS)} dims):")
    for key, (lo, hi) in zip(PARAM_KEYS, bounds):
        print(f"  {key:35s}  [{lo:.3g}, {hi:.3g}]")

    bo = BayesianOptimizer(
        f=lambda x: objective(x, ref),
        bounds=bounds,
        n_initial_points=N_INITIAL_POINTS,
        iter_limit=BO_ITER_LIMIT,
        n_restarts=5,
        seed=42,
    )

    best_x, best_score = bo.run()

    best_params = params_from_x(best_x, ref)
    print("\n[BO] Optimisation complete.")
    print(f"[BO] Best score: {best_score:.4f}")
    print("[BO] Best parameters:")
    for k, v in best_params.items():
        print(f"  {k:35s}: {v}")

    out_path = Path(__file__).parent.parent / "src/tmp/planning/NH3-Chile-bo.yml"
    with open(out_path, "w") as f:
        yaml.dump(best_params, f, default_flow_style=False)
    print(f"[BO] Best parameters saved to {out_path}")

    return best_params, best_score


if __name__ == "__main__":
    run_bayesopt()
