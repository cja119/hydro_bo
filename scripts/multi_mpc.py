"""
Run RayMultiMPC over a planning model and log results to disk.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
import os
import time as _time
import yaml
import numpy as np
from datetime import datetime

from hydro_bo.utils.logging_config import configure_logging, get_logger
from hydro_bo.mpc.dispatcher import RayMultiMPC
from hydro_bo.utils.seeding import resolve_master_seed

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VECTOR          = "NH3"
PLANNING_MODEL  = f"{VECTOR}-Chile.yml"
WEATHER_FILE    = ["CoastalChile_05-10_Wind.csv", "CoastalChile_10-15_Wind.csv", "CoastalChile_15-20_Wind.csv", "CoastalChile_20-21_Wind.csv", "CoastalChile_21-22_Wind.csv", "CoastalChile_23-24_Wind.csv"]

NUM_INSTANCES   = os.cpu_count() - 1  # total MPC runs
NUM_DEVICES     = os.cpu_count() - 1  # simultaneous workers
TIMEOUT         = 900                  # seconds per worker

DUMP_DIAGNOSTICS_ON_FAILURE = False


def load_planning_model(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_multisim(args=None):
    now = datetime.now()
    run_timestamp = now.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).parent / "tmp" / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S") / VECTOR
    out_dir.mkdir(parents=True, exist_ok=True)

    configure_logging(log_file=out_dir / "run.log")

    cli_seed = getattr(args, "master_seed", None) if args is not None else None
    master_seed = resolve_master_seed(cli_seed)
    logger.info("multi_mpc.master_seed", master_seed=master_seed, cli_seed=cli_seed)

    if args is not None:
        args_dict = vars(args)
        args_dict["master_seed_resolved"] = master_seed
        with open(out_dir / "args.json", "w") as f:
            json.dump(args_dict, f, indent=2)
        logger.info("multi_mpc.args_saved", path=str(out_dir / "args.json"))

    planning_model_path = Path(__file__).parent / "tmp/planning" / PLANNING_MODEL
    if not planning_model_path.exists():
        logger.error(
            "multi_mpc.missing_planning_model",
            path=str(planning_model_path),
            message=f"Planning model not found. Run: python scripts/planning.py {VECTOR}",
        )
        return

    params = load_planning_model(planning_model_path)
    logger.info("multi_mpc.loaded_model", path=str(planning_model_path))

    env_args = {
        "config": {
            "vector": VECTOR,
            "mpc": {"planning_model": PLANNING_MODEL},
            "weather_data": {"weather_file": WEATHER_FILE},
            "price_dynamics": {"enabled": args.dynamic_price if args is not None else False},
        },
    }

    dispatcher = RayMultiMPC(
        env_args=env_args,
        param_overrides=params,
        num_instances=NUM_INSTANCES,
        num_devices=NUM_DEVICES,
        timeout=TIMEOUT,
        exit_fraction=1.0,
        dump_diagnostics_on_failure=DUMP_DIAGNOSTICS_ON_FAILURE,
        log_dir=out_dir,
        master_seed=master_seed,
    )

    logger.info("multi_mpc.initialized", num_instances=NUM_INSTANCES, num_devices=NUM_DEVICES, timeout=TIMEOUT)

    walltime_seconds = getattr(args, "walltime_seconds", None) if args is not None else None
    buffer_seconds = getattr(args, "buffer_seconds", 300) if args is not None else 300
    deadline = None
    if walltime_seconds is not None:
        deadline = _time.perf_counter() + walltime_seconds - buffer_seconds
        logger.info("multi_mpc.deadline_set",
                    walltime_seconds=walltime_seconds, buffer_seconds=buffer_seconds)

    scores = dispatcher.run_multisim(deadline=deadline)

    scores_arr = np.array(scores, dtype=float) if scores else np.array([], dtype=float)
    finite_mask = np.isfinite(scores_arr)
    finite_scores = scores_arr[finite_mask]
    n_failed = int((~finite_mask).sum())

    if len(finite_scores) > 0:
        mean_score = float(finite_scores.mean())
        var_score  = float(finite_scores.var())
    else:
        mean_score = float("nan")
        var_score  = float("nan")

    logger.info("multi_mpc.complete",
                n_workers=len(scores),
                mean_score=mean_score,
                var_score=var_score,
                n_failed=n_failed,
                scores=scores)

    # Save results
    result = {
        "timestamp": now.isoformat(),
        "vector": VECTOR,
        "planning_model": PLANNING_MODEL,
        "num_instances": NUM_INSTANCES,
        "num_devices": NUM_DEVICES,
        "dynamic_price": args.dynamic_price if args is not None else False,
        "n_workers": len(scores),
        "n_failed": n_failed,
        "mean_score": mean_score,
        "var_score": var_score,
        "worker_scores": scores,
    }

    json_path = out_dir / f"results_{run_timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("multi_mpc.results_saved", path=str(json_path))

    csv_path = out_dir / f"results_{run_timestamp}.csv"
    fieldnames = ["timestamp", "vector", "planning_model", "num_instances", "num_devices",
                  "dynamic_price", "n_workers", "mean_score", "var_score"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({k: result[k] for k in fieldnames})
    logger.info("multi_mpc.csv_saved", path=str(csv_path))

    import matplotlib.pyplot as plt
    plt.hist(scores, bins=10)
    plt.title(f"Distribution of MPC Scores — {VECTOR}")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plot_path = out_dir / f"scores_{run_timestamp}.png"
    plt.savefig(plot_path)
    logger.info("multi_mpc.plot_saved", path=str(plot_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RayMultiMPC over a planning model.")
    parser.add_argument("--vector",        type=str, default=VECTOR,
                        help="Hydrogen vector (default: %(default)s)")
    parser.add_argument("--ncpus",         type=int, default=NUM_DEVICES,
                        help="Number of parallel MPC workers (default: %(default)s)")
    parser.add_argument("--n_sim",         type=int, default=None,
                        help="Total MPC runs. Defaults to --ncpus if not given.")
    parser.add_argument("--dynamic_price", type=lambda x: x.lower() in ("true", "1", "yes"), default=False,
                        help="Enable OU + jump-diffusion hydrogen price dynamics: True/False (default: False).")
    parser.add_argument("--walltime_seconds", type=int, default=None,
                        help="Total PBS walltime in seconds. Enables deadline-based early exit.")
    parser.add_argument("--buffer_seconds",   type=int, default=300,
                        help="Seconds to reserve before walltime for result saving (default: 300).")
    parser.add_argument("--master_seed",      type=int, default=None,
                        help="Master seed for the run. If omitted, derived from PBS env vars + pid + wall time.")
    parser.add_argument("--planning_model",   type=str, default=None,
                        help="Planning-model filename under tmp/planning/ to load instead of the default "
                             "<VECTOR>-Chile.yml. Used to inject specific param sets (e.g. a sobol row).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    VECTOR          = args.vector
    NUM_DEVICES     = args.ncpus
    NUM_INSTANCES   = args.n_sim if args.n_sim is not None else args.ncpus

    PLANNING_MODEL  = args.planning_model or f"{VECTOR}-Chile.yml"

    run_multisim(args=args)
