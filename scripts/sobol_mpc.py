"""
Run a single Sobol-sampled MPC evaluation.

Loads a row from sobol_indices.csv (unit-hypercube samples), scales it to
parameter bounds derived from the reference planning model, then runs
RayMultiMPC and records the objective (mean - λ·var).

Designed to be submitted as a PBS array job via jobs/sobol.sh, with each
array element setting --index_row to its PBS_ARRAY_INDEX.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # import from bayesopt

import argparse
import csv
import json
import os
import time as _time
import numpy as np
from datetime import datetime

from hydro_bo.algs.logging_config import configure_logging, get_logger
from hydro_bo.algs.dispatcher import RayMultiMPC

# Shared helpers — import from bayesopt to avoid duplication
from bayesopt import (
    PARAM_KEYS,
    RENEWABLES,
    STDEV_PENALTY,
    build_bounds,
    params_from_x,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VECTOR               = "NH3"
PLANNING_MODEL       = "NH3-Chile.yml"
WEATHER_FILE         = "CoastalChile_15-20_Wind.csv"

NUM_INSTANCES        = os.cpu_count() - 1
NUM_DEVICES          = os.cpu_count() - 1
TIMEOUT              = 900

SOBOL_CSV            = Path(__file__).parent / "sobol_indices.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_reference(path: Path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def load_sobol_row(csv_path: Path, index_row: int) -> np.ndarray:
    """Return the unit-hypercube sample at the given row (0-indexed, header excluded)."""
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if header != PARAM_KEYS:
            raise ValueError(
                f"sobol_indices.csv columns {header} do not match PARAM_KEYS {PARAM_KEYS}"
            )
        for i, row in enumerate(reader):
            if i == index_row:
                return np.array([float(v) for v in row])
    raise IndexError(
        f"index_row={index_row} is out of range for {csv_path} "
        f"(file has {i + 1} data rows)"
    )


def scale_unit_to_bounds(unit: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Scale a [0,1) sample to the parameter bounds."""
    lo, hi = bounds[:, 0], bounds[:, 1]
    return lo + unit * (hi - lo)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_sobol_eval(args=None):
    now = datetime.now()
    run_timestamp = now.strftime("%Y%m%d_%H%M%S")

    # Each row gets its own subdirectory so parallel jobs never collide.
    # Directory is stable across days so multi-day runs land in the same place.
    index_row = args.index_row if args is not None else 0
    dynamic_price = args.dynamic_price if args is not None else False
    run_label = f"{VECTOR}-dynamic_price" if dynamic_price else VECTOR
    out_dir = (
        Path(__file__).parent
        / "tmp"
        / "sobol"
        / run_label
        / f"row_{index_row:05d}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    configure_logging(log_file=out_dir / "run.log")

    if args is not None:
        with open(out_dir / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2)
        logger.info("sobol_mpc.args_saved", path=str(out_dir / "args.json"))

    # ------------------------------------------------------------------
    # Load reference and build bounds
    # ------------------------------------------------------------------
    planning_model_path = Path(__file__).parent / "tmp/planning" / PLANNING_MODEL
    if not planning_model_path.exists():
        logger.error(
            "sobol_mpc.missing_planning_model",
            path=str(planning_model_path),
            message=f"Run: python scripts/planning.py {VECTOR}",
        )
        return

    ref = load_reference(planning_model_path)
    bounds_expansion = args.bounds_expansion if args is not None else 0.5
    bounds = build_bounds(ref, bounds_expansion)

    logger.info("sobol_mpc.bounds_built", bounds_expansion=bounds_expansion)
    for key, (lo, hi) in zip(PARAM_KEYS, bounds):
        logger.info("sobol_mpc.param_bounds", parameter=key, lower=lo, upper=hi)

    # ------------------------------------------------------------------
    # Load Sobol row and scale to bounds
    # ------------------------------------------------------------------
    if not SOBOL_CSV.exists():
        logger.error("sobol_mpc.missing_csv", path=str(SOBOL_CSV))
        return

    unit_sample = load_sobol_row(SOBOL_CSV, index_row)
    x = scale_unit_to_bounds(unit_sample, bounds)

    logger.info("sobol_mpc.sample_loaded", index_row=index_row, unit_sample=unit_sample.tolist())
    logger.info("sobol_mpc.sample_scaled", x=x.tolist())

    params = params_from_x(x, ref)
    logger.info("sobol_mpc.parameters", **{k: params[k] for k in PARAM_KEYS})

    # ------------------------------------------------------------------
    # Run multisim
    # ------------------------------------------------------------------
    env_args = {
        "config": {
            "vector": VECTOR,
            "mpc": {"planning_model": PLANNING_MODEL},
            "weather_data": {"weather_file": WEATHER_FILE},
            "price_dynamics": {"enabled": dynamic_price},
        },
    }

    dispatcher = RayMultiMPC(
        env_args=env_args,
        param_overrides=params,
        num_instances=NUM_INSTANCES,
        num_devices=NUM_DEVICES,
        timeout=TIMEOUT,
        exit_fraction=1.0,
    )

    # Compute deadline from walltime budget if provided
    walltime_seconds = getattr(args, "walltime_seconds", None) if args is not None else None
    buffer_seconds = getattr(args, "buffer_seconds", 300) if args is not None else 300
    deadline = None
    if walltime_seconds is not None:
        deadline = _time.perf_counter() + walltime_seconds - buffer_seconds
        logger.info("sobol_mpc.deadline_set",
                    walltime_seconds=walltime_seconds, buffer_seconds=buffer_seconds)

    logger.info("sobol_mpc.dispatcher_initialized",
                num_instances=NUM_INSTANCES, num_devices=NUM_DEVICES, timeout=TIMEOUT)

    scores = dispatcher.run_multisim(deadline=deadline)

    scores_arr = np.array(scores, dtype=float) if scores else np.array([], dtype=float)
    finite_mask = np.isfinite(scores_arr)
    finite_scores = scores_arr[finite_mask]
    n_failed = int((~finite_mask).sum())

    if len(finite_scores) > 0:
        mean_score = float(finite_scores.mean())
        var_score  = float(finite_scores.var(ddof=1)) if len(finite_scores) >= 2 else float("nan")
        sd_score   = float(np.sqrt(var_score)) if np.isfinite(var_score) else 0.0
        objective  = mean_score - STDEV_PENALTY * sd_score
    else:
        mean_score = float("nan")
        var_score  = float("nan")
        objective  = -1e8

    logger.info("sobol_mpc.complete",
                index_row=index_row,
                objective=objective,
                mean_score=mean_score,
                var_score=var_score,
                n_failed=n_failed,
                n_workers=len(scores))

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    status = "complete" if walltime_seconds is None or n_failed == 0 else "partial"
    result = {
        "timestamp": now.isoformat(),
        "status": status,
        "index_row": index_row,
        "vector": VECTOR,
        "planning_model": PLANNING_MODEL,
        "bounds_expansion": bounds_expansion,
        "dynamic_price": dynamic_price,
        "num_instances": NUM_INSTANCES,
        "num_devices": NUM_DEVICES,
        "objective": objective,
        "mean_score": mean_score,
        "var_score": var_score,
        "n_workers": len(scores),
        "n_failed": n_failed,
        "worker_scores": scores,
        "unit_sample": unit_sample.tolist(),
        "x": x.tolist(),
    }
    for key in PARAM_KEYS:
        result[key] = params[key]
    result["capex"] = params["capex"]
    result["opex"]  = params["opex"]

    json_path = out_dir / f"result_{run_timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("sobol_mpc.result_saved", path=str(json_path))

    csv_fieldnames = (
        ["timestamp", "index_row", "vector", "bounds_expansion", "dynamic_price",
         "num_instances", "num_devices", "objective", "mean_score", "var_score", "n_workers"]
        + PARAM_KEYS
        + ["capex", "opex"]
    )
    csv_path = out_dir / f"result_{run_timestamp}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(result)
    logger.info("sobol_mpc.csv_saved", path=str(csv_path))


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single Sobol-sampled MPC evaluation (designed for PBS array jobs)."
    )
    parser.add_argument("--vector",          type=str,   default=VECTOR,
                        help="Hydrogen vector (default: %(default)s)")
    parser.add_argument("--ncpus",           type=int,   default=NUM_DEVICES,
                        help="Number of parallel MPC workers (default: %(default)s)")
    parser.add_argument("--n_sim",           type=int,   default=None,
                        help="Total MPC runs. Defaults to --ncpus if not given.")
    parser.add_argument("--bounds_expansion",type=float, default=0.5,
                        help="±fractional expansion around reference values (default: %(default)s)")
    parser.add_argument("--index_row",       type=int,   required=True,
                        help="Row index into sobol_indices.csv (0-based, header excluded).")
    parser.add_argument("--dynamic_price",   type=lambda x: x.lower() in ("true", "1", "yes"),
                        default=False,
                        help="Enable OU + jump-diffusion price dynamics: True/False (default: False).")
    parser.add_argument("--walltime_seconds", type=int, default=None,
                        help="Total PBS walltime in seconds. Enables deadline-based early exit.")
    parser.add_argument("--buffer_seconds",   type=int, default=300,
                        help="Seconds to reserve before walltime for result saving (default: 300).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    VECTOR         = args.vector
    NUM_DEVICES    = args.ncpus
    NUM_INSTANCES  = args.n_sim if args.n_sim is not None else args.ncpus
    PLANNING_MODEL = f"{VECTOR}-Chile.yml"

    run_sobol_eval(args=args)
