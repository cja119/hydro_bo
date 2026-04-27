"""
Bayesian optimisation over the MPC planning model parameters.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import os
import yaml
import numpy as np
import csv
from datetime import datetime

from hydro_bo.algs.logging_config import configure_logging, get_logger
from hydro_bo.algs.dispatcher import RayMultiMPC
from hydro_bo.algs.bayesopt import BayesianOptimizer, configure_jax_threads
from hydro_bo.algs.seeding import resolve_master_seed
from hydro_bo.envs.shipping.utils import calculate_capex_opex

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PLANNING_MODEL   = "NH3-Chile.yml"
VECTOR           = "NH3"
RENEWABLES       = "wind"
WEATHER_FILE     = ["CoastalChile_05-10_Wind.csv", "CoastalChile_10-15_Wind.csv", "CoastalChile_15-20_Wind.csv", "CoastalChile_20-21_Wind.csv", "CoastalChile_21-22_Wind.csv", "CoastalChile_23-24_Wind.csv"]
REFERENCE_PATH   = Path(__file__).parent / "tmp/planning" / PLANNING_MODEL

BOUNDS_EXPANSION  = 0.5         # ±50 % around reference values
STDEV_PENALTY     = 0.5         # λ in noisy-EI on g(x) = mu(x) - λ·sigma(x)
N_INITIAL_POINTS  = 8           # Sobol evaluations before BO starts
BO_ITER_LIMIT     = 20          # BO iterations
N_INSTANCES       = 15          # total MPC runs per BO evaluation (parallelised across NUM_DEVICES)
NUM_DEVICES       = os.cpu_count() - 1  # simultaneous instances
TIMEOUT           = 900         # seconds per evaluation
MIN_VALID_SAMPLES = 4           # minimum valid worker scores per eval; below → penalty
FAILURE_PENALTY   = -10.0       # synthetic worst-case sample injected when too few valid scores

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

# Global counter, results storage, and output directory for incremental saving
_eval_counter = 0
_results_log = []
_bayesopt_dir: Path | None = None
_run_timestamp: str = ""
_master_seed: int = 0

def objective(x: np.ndarray, ref: dict) -> np.ndarray:
    """Run N_INSTANCES MPC simulations at x and return the array of valid
    worker scores. Drops non-finite entries and dispatcher penalty values
    (≤ -1e7). If fewer than MIN_VALID_SAMPLES survive, returns a single
    synthetic FAILURE_PENALTY sample so the surrogate learns to avoid x."""
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
        master_seed=_master_seed + _eval_counter,
    )

    raw_scores = dispatcher.run_multisim()
    arr = np.asarray(raw_scores, dtype=float).ravel() if raw_scores else np.array([], dtype=float)
    valid_mask = np.isfinite(arr) & (arr > -1e7)
    valid_scores = arr[valid_mask]
    n_valid = int(valid_scores.size)
    n_failed = int(arr.size - n_valid)

    if n_valid >= MIN_VALID_SAMPLES:
        bo_samples = valid_scores
        mean_score = float(valid_scores.mean())
        var_score = float(valid_scores.var(ddof=1)) if n_valid >= 2 else float("nan")
        sd_score = float(np.sqrt(var_score)) if np.isfinite(var_score) else 0.0
        objective_value = mean_score - STDEV_PENALTY * sd_score
        penalty_applied = False
    else:
        bo_samples = np.array([FAILURE_PENALTY], dtype=float)
        mean_score = FAILURE_PENALTY
        var_score = float("nan")
        objective_value = FAILURE_PENALTY
        penalty_applied = True

    result_entry = {
        'eval_id': _eval_counter,
        'timestamp': datetime.now().isoformat(),
        'objective': objective_value,
        'mean_score': mean_score,
        'var_score': var_score,
        'num_workers': len(raw_scores),
        'n_valid': n_valid,
        'n_failed': n_failed,
        'penalty_applied': penalty_applied,
        'worker_scores': list(raw_scores),
    }
    for key in PARAM_KEYS:
        result_entry[key] = params[key]
    result_entry['capex'] = params['capex']
    result_entry['opex'] = params['opex']

    _results_log.append(result_entry)

    logger.info("bayesopt.evaluation",
               eval_id=_eval_counter,
               objective=objective_value,
               mean=mean_score,
               variance=var_score,
               n_valid=n_valid,
               n_failed=n_failed,
               penalty_applied=penalty_applied)

    if _bayesopt_dir is not None:
        save_results_to_files(_results_log, _bayesopt_dir, _run_timestamp)

    return bo_samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_sobol_cache(sobol_dir: Path, expected_vector: str,
                     expected_bounds_expansion: float,
                     expected_dynamic_price: bool) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load cached (x, valid_worker_scores) pairs from a sobol_mpc.py results directory.

    Filters non-finite and dispatcher-penalty (≤ -1e7) worker scores. Rows
    with fewer than MIN_VALID_SAMPLES survivors are dropped entirely (they
    don't seed the GP — the BO will treat that x as unobserved)."""
    observations: list[tuple[np.ndarray, np.ndarray]] = []
    n_missing_result = 0
    n_mismatched = 0
    n_failed = 0

    row_dirs = sorted(sobol_dir.glob("row_*"))
    for rd in row_dirs:
        result_files = sorted(rd.glob("result_*.json"))
        if not result_files:
            n_missing_result += 1
            continue
        with open(result_files[-1]) as f:
            data = json.load(f)

        if data.get("vector") != expected_vector:
            n_mismatched += 1
            continue
        if float(data.get("bounds_expansion", -1)) != float(expected_bounds_expansion):
            n_mismatched += 1
            continue
        if bool(data.get("dynamic_price", False)) != bool(expected_dynamic_price):
            n_mismatched += 1
            continue

        scores = data.get("worker_scores")
        if scores is None:
            obj = data.get("objective")
            if obj is None or not np.isfinite(obj) or obj <= -1e7:
                n_failed += 1
                continue
            scores = [obj]

        arr = np.asarray(scores, dtype=float).ravel()
        valid = np.isfinite(arr) & (arr > -1e7)
        valid_arr = arr[valid]
        if valid_arr.size < MIN_VALID_SAMPLES:
            n_failed += 1
            continue

        observations.append((np.asarray(data["x"], dtype=float), valid_arr))

    logger.info("bayesopt.sobol_cache_loaded",
                dir=str(sobol_dir),
                n_rows_found=len(row_dirs),
                n_loaded=len(observations),
                n_missing_result=n_missing_result,
                n_mismatched=n_mismatched,
                n_failed=n_failed)
    return observations


def save_results_to_files(results_log: list, bayesopt_dir: Path, run_timestamp: str):
    """Save results to CSV and JSON, overwriting on each call for crash-safety."""
    if not results_log:
        return

    # Save summary CSV (one row per evaluation, no worker_scores column)
    fieldnames = ['eval_id', 'timestamp', 'objective', 'mean_score', 'var_score',
                  'num_workers', 'n_valid', 'n_failed', 'penalty_applied']
    fieldnames.extend(PARAM_KEYS)
    fieldnames.extend(['capex', 'opex'])

    csv_path = bayesopt_dir / f"bo_results_{run_timestamp}.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for entry in results_log:
            writer.writerow({k: v for k, v in entry.items() if k != 'worker_scores'})

    # Save detailed JSON (includes individual worker scores)
    json_path = bayesopt_dir / f"bo_results_detailed_{run_timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results_log, f, indent=2)


def run_bayesopt(args=None):
    global _eval_counter, _results_log, _bayesopt_dir, _run_timestamp, _master_seed
    _eval_counter = 0
    _results_log = []

    now = datetime.now()
    _run_timestamp = now.strftime("%Y%m%d_%H%M%S")
    _bayesopt_dir = Path(__file__).parent / "tmp" / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S") / VECTOR
    _bayesopt_dir.mkdir(parents=True, exist_ok=True)

    configure_logging(log_file=_bayesopt_dir / "run.log")

    cli_seed = getattr(args, "master_seed", None) if args is not None else None
    _master_seed = resolve_master_seed(cli_seed)
    logger.info("bayesopt.master_seed", master_seed=_master_seed, cli_seed=cli_seed)

    if args is not None:
        args_dict = vars(args)
        args_dict["n_sim_resolved"] = N_INSTANCES  # record the resolved value
        args_dict["master_seed_resolved"] = _master_seed
        with open(_bayesopt_dir / "args.json", "w") as f:
            json.dump(args_dict, f, indent=2)
        logger.info("bayesopt.args_saved", path=str(_bayesopt_dir / "args.json"))

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
        lam=STDEV_PENALTY,
        n_restarts=5,
        seed=_master_seed % (2**31),
    )

    sobol_dir_arg = getattr(args, "sobol_dir", None) if args is not None else None
    if sobol_dir_arg:
        sobol_dir = Path(sobol_dir_arg)
        if not sobol_dir.is_absolute():
            sobol_dir = Path(__file__).parent / sobol_dir
        if not sobol_dir.exists():
            logger.error("bayesopt.sobol_dir_missing", path=str(sobol_dir))
        else:
            preloaded = load_sobol_cache(
                sobol_dir,
                expected_vector=VECTOR,
                expected_bounds_expansion=BOUNDS_EXPANSION,
                expected_dynamic_price=args.dynamic_price,
            )
            for x, samples in preloaded:
                bo.observe(x, samples)
            if preloaded:
                bo.n_initial_points = len(preloaded)
                logger.info("bayesopt.sobol_phase_skipped_via_cache",
                            n_preloaded=len(preloaded))

    best_x, best_score = bo.run()

    best_params = params_from_x(best_x, ref)
    logger.info("bayesopt.complete")
    logger.info("bayesopt.best_score", score=best_score)
    logger.info("bayesopt.best_parameters")
    for k, v in best_params.items():
        logger.info("bayesopt.parameter", name=k, value=v)

    out_path = Path(__file__).parent / "tmp" / "planning" / f"{VECTOR}-Chile-bo.yml"
    with open(out_path, "w") as f:
        yaml.dump(best_params, f, default_flow_style=False)
    logger.info("bayesopt.saved", path=str(out_path))

    # Final save (incremental saves already happened after each eval)
    save_results_to_files(_results_log, _bayesopt_dir, _run_timestamp)

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
    parser.add_argument("--dynamic_price", type=lambda x: x.lower() in ("true", "1", "yes"), default=False,
                        help="Enable OU + jump-diffusion hydrogen price dynamics: True/False (default: False).")
    parser.add_argument("--sobol_dir",     type=str,   default=None,
                        help="Path to a sobol_mpc.py results directory (e.g. tmp/sobol/NH3/). "
                             "If set, cached (x, objective) pairs matching --vector, --scale_factor, "
                             "and --dynamic_price seed the GP and the Sobol phase is skipped.")
    parser.add_argument("--master_seed",   type=int,   default=None,
                        help="Master seed. If omitted, derived from PBS env vars + pid + wall time.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Parallelise GP fits across the same core budget Ray uses for workers.
    # Must run before any jax import (the bayesopt module imports jax lazily,
    # so this is still safe here).
    configure_jax_threads(args.ncpus)

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
            "price_dynamics": {"enabled": args.dynamic_price},
        },
    }

    run_bayesopt(args=args)
