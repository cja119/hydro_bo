"""
Bayesian optimisation over the MPC planning model parameters.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import os
import yaml
import numpy as np
import csv
from datetime import datetime

from hydro_bo import configure_logging
from hydro_bo.algs.logging_config import get_logger
from hydro_bo.algs.dispatcher import RayMultiMPC
from hydro_bo.algs.bayesopt import BayesianOptimizer
from hydro_bo.envs.shipping.utils import calculate_capex_opex

configure_logging()
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PLANNING_MODEL   = "NH3-Chile.yml"
VECTOR           = "NH3"
RENEWABLES       = "wind"
WEATHER_FILE     = "CoastalChile_15-20_Wind.csv"
REFERENCE_PATH   = Path(__file__).parent / "tmp/planning" / PLANNING_MODEL

BOUNDS_EXPANSION = 0.5         # ±50 % around reference values
VARIANCE_PENALTY = 0.5         # λ in: mean - λ * var
N_INITIAL_POINTS = 8           # Sobol evaluations before BO starts
BO_ITER_LIMIT    = 20          # BO iterations
N_INSTANCES      = 15           # total MPC runs per BO evaluation (parallelised across NUM_DEVICES)
NUM_DEVICES      = os.cpu_count() - 1  # simultaneous instances
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

# Global counter and results storage for logging
_eval_counter = 0
_results_log = []

def objective(x: np.ndarray, ref: dict) -> float:
    """Evaluate mean - λ·var over N_INSTANCES MPC runs for parameter vector x."""
    global _eval_counter
    _eval_counter += 1

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
        objective_value = -np.inf
    else:
        scores_arr = np.array(scores, dtype=float)
        mean_score = float(scores_arr.mean())
        var_score = float(scores_arr.var())
        objective_value = mean_score - VARIANCE_PENALTY * var_score

        # Log this evaluation with individual worker scores
        result_entry = {
            'eval_id': _eval_counter,
            'timestamp': datetime.now().isoformat(),
            'objective': objective_value,
            'mean_score': mean_score,
            'var_score': var_score,
            'num_workers': len(scores),
            'worker_scores': scores,
        }

        # Add all parameters to the log entry
        for i, key in enumerate(PARAM_KEYS):
            result_entry[key] = params[key]

        result_entry['capex'] = params['capex']
        result_entry['opex'] = params['opex']

        _results_log.append(result_entry)

        logger.info("bayesopt.evaluation",
                   eval_id=_eval_counter,
                   objective=objective_value,
                   mean=mean_score,
                   variance=var_score,
                   n_workers=len(scores))

    return objective_value


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def save_results_to_files(results_log: list, bayesopt_dir: Path, run_timestamp: str):
    """Save results to both CSV and JSON files."""
    # Create the bayesopt directory
    bayesopt_dir.mkdir(parents=True, exist_ok=True)

    # Save summary CSV (one row per evaluation)
    csv_path = bayesopt_dir / f"bo_results_{run_timestamp}.csv"
    with open(csv_path, 'w', newline='') as f:
        if not results_log:
            return

        # Define CSV columns
        fieldnames = ['eval_id', 'timestamp', 'objective', 'mean_score', 'var_score', 'num_workers']
        fieldnames.extend(PARAM_KEYS)
        fieldnames.extend(['capex', 'opex'])

        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()

        for entry in results_log:
            # Create a copy without worker_scores for CSV
            csv_entry = {k: v for k, v in entry.items() if k != 'worker_scores'}
            writer.writerow(csv_entry)

    logger.info("bayesopt.csv_saved", path=str(csv_path), n_evaluations=len(results_log))

    # Save detailed JSON (includes individual worker scores)
    json_path = bayesopt_dir / f"bo_results_detailed_{run_timestamp}.json"
    import json
    with open(json_path, 'w') as f:
        json.dump(results_log, f, indent=2)

    logger.info("bayesopt.json_saved", path=str(json_path))


def run_bayesopt():
    global _eval_counter, _results_log
    _eval_counter = 0
    _results_log = []

    now = datetime.now()
    run_timestamp = now.strftime("%Y%m%d_%H%M%S")
    bayesopt_dir = Path(__file__).parent / "tmp" / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S") / VECTOR

    try:
        ref = load_reference(REFERENCE_PATH)
    except FileNotFoundError:
        logger.error(
            "bayesopt.missing_planning_model",
            path=str(REFERENCE_PATH),
            message=f"Planning model file not found. Please run the planning model first: python scripts/planning.py {VECTOR}"
        )
        return None, None

    bounds = build_bounds(ref, BOUNDS_EXPANSION)

    logger.info("bayesopt.search_space", dims=len(PARAM_KEYS))
    for key, (lo, hi) in zip(PARAM_KEYS, bounds):
        logger.info("bayesopt.parameter_bounds", parameter=key, lower=lo, upper=hi)

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
    logger.info("bayesopt.complete")
    logger.info("bayesopt.best_score", score=best_score)
    logger.info("bayesopt.best_parameters")
    for k, v in best_params.items():
        logger.info("bayesopt.parameter", name=k, value=v)

    out_path = Path(__file__).parent / "tmp/planning/NH3-Chile-bo.yml"
    with open(out_path, "w") as f:
        yaml.dump(best_params, f, default_flow_style=False)
    logger.info("bayesopt.saved", path=str(out_path))

    # Save all evaluation results
    save_results_to_files(_results_log, bayesopt_dir, run_timestamp)

    return best_params, best_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bayesian optimisation over MPC planning model parameters.")
    parser.add_argument("--vector",       type=str,   default=VECTOR,           help="Hydrogen vector (default: %(default)s)")
    parser.add_argument("--iter_budget",  type=int,   default=BO_ITER_LIMIT,    help="Number of BO iterations (default: %(default)s)")
    parser.add_argument("--scale_factor", type=float, default=BOUNDS_EXPANSION, help="±bounds expansion fraction (default: %(default)s)")
    parser.add_argument("--ncpus",        type=int,   default=NUM_DEVICES,      help="Number of parallel MPC workers (default: %(default)s)")
    parser.add_argument("--nsobol",       type=int,   default=N_INITIAL_POINTS, help="Sobol evaluations before BO starts (default: %(default)s)")
    parser.add_argument("--n_sim",        type=int,   default=None,
                        help="Total MPC runs per BO evaluation. Defaults to --ncpus (one per CPU) if not given.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Override module-level constants from CLI arguments
    VECTOR           = args.vector
    BO_ITER_LIMIT    = args.iter_budget
    BOUNDS_EXPANSION = args.scale_factor
    NUM_DEVICES      = args.ncpus
    N_INITIAL_POINTS = args.nsobol
    N_INSTANCES      = args.n_sim if args.n_sim is not None else args.ncpus

    # Derive vector-dependent paths/config from the (possibly overridden) VECTOR
    PLANNING_MODEL = f"{VECTOR}-Chile.yml"
    REFERENCE_PATH = Path(__file__).parent / "tmp/planning" / PLANNING_MODEL
    ENV_ARGS = {
        "config": {
            "vector": VECTOR,
            "mpc": {"planning_model": PLANNING_MODEL},
            "weather_data": {"weather_file": WEATHER_FILE},
        },
    }

    run_bayesopt()
