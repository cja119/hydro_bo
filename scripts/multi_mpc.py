"""
Run RayMultiMPC over a planning model and log results to disk.

Reads `scripts/config.yml` for the run knobs (vector, weather_file,
forecast_horizon, num_instances, num_devices, timeout, dynamic_price,
log_level). CLI flags override individual fields where given.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
import time as _time
import yaml
import numpy as np
from datetime import datetime

from hydro_bo.utils.logging_config import configure_logging, get_logger
from hydro_bo.utils.jax_threads import configure_jax_threads
from hydro_bo.utils.run_config import Config, env_args_from, load_config
from hydro_bo.utils.seeding import resolve_master_seed

logger = get_logger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parent

DUMP_DIAGNOSTICS_ON_FAILURE = False


def load_planning_model(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_multisim(cfg: Config, args: argparse.Namespace):
    g = cfg.general
    now = datetime.now()
    run_timestamp = now.strftime("%Y%m%d_%H%M%S")
    out_dir = SCRIPTS_DIR / "tmp" / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S") / g.vector
    out_dir.mkdir(parents=True, exist_ok=True)

    configure_logging(log_file=out_dir / "run.log", package_level=g.log_level)

    cli_seed = getattr(args, "master_seed", None)
    master_seed = resolve_master_seed(cli_seed)
    logger.info("multi_mpc.master_seed", master_seed=master_seed, cli_seed=cli_seed)

    # Persist resolved config + CLI args for traceability.
    snapshot = {
        "general": g.__dict__,
        "args": vars(args),
        "master_seed_resolved": master_seed,
    }
    with open(out_dir / "args.json", "w") as f:
        json.dump(snapshot, f, indent=2, default=str)
    logger.info("multi_mpc.args_saved", path=str(out_dir / "args.json"))

    planning_model_filename = args.planning_model or f"{g.vector}-Chile.yml"
    planning_model_path = SCRIPTS_DIR / "tmp" / "planning" / planning_model_filename
    if not planning_model_path.exists():
        logger.error(
            "multi_mpc.missing_planning_model",
            path=str(planning_model_path),
            message=f"Planning model not found. Run: python scripts/planning.py {g.vector}",
        )
        return

    params = load_planning_model(planning_model_path)
    logger.info("multi_mpc.loaded_model", path=str(planning_model_path))

    # env_args_from(cfg) populates "Time.forecast_horizon" from
    # `general.forecast_horizon`, plus weather files, price-dynamics enable
    # and vector. Apply per-CLI overrides on top.
    env_args = env_args_from(cfg)
    env_args["config"]["mpc"]["planning_model"] = planning_model_filename
    if args.dynamic_price is not None:
        env_args["config"]["price_dynamics"]["enabled"] = bool(args.dynamic_price)

    num_instances = args.n_sim if args.n_sim is not None else g.num_instances
    num_devices = g.num_devices

    from hydro_bo.mpc.dispatcher import RayMultiMPC  # noqa: E402  defer until JAX threads are configured

    dispatcher = RayMultiMPC(
        env_args=env_args,
        param_overrides=params,
        num_instances=num_instances,
        num_devices=num_devices,
        timeout=g.timeout,
        exit_fraction=1.0,
        dump_diagnostics_on_failure=DUMP_DIAGNOSTICS_ON_FAILURE,
        log_dir=out_dir,
        master_seed=master_seed,
    )

    logger.info(
        "multi_mpc.initialized",
        num_instances=num_instances,
        num_devices=num_devices,
        timeout=g.timeout,
        forecast_horizon=g.forecast_horizon,
        dynamic_price=env_args["config"]["price_dynamics"]["enabled"],
    )

    walltime_seconds = getattr(args, "walltime_seconds", None)
    buffer_seconds = getattr(args, "buffer_seconds", 300)
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

    result = {
        "timestamp": now.isoformat(),
        "vector": g.vector,
        "planning_model": planning_model_filename,
        "forecast_horizon": g.forecast_horizon,
        "num_instances": num_instances,
        "num_devices": num_devices,
        "dynamic_price": env_args["config"]["price_dynamics"]["enabled"],
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
    fieldnames = ["timestamp", "vector", "planning_model", "forecast_horizon",
                  "num_instances", "num_devices", "dynamic_price", "n_workers",
                  "n_failed", "mean_score", "var_score"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({k: result[k] for k in fieldnames})
    logger.info("multi_mpc.csv_saved", path=str(csv_path))

    import matplotlib.pyplot as plt
    plt.hist(scores, bins=10)
    plt.title(f"Distribution of MPC Scores — {g.vector}")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plot_path = out_dir / f"scores_{run_timestamp}.png"
    plt.savefig(plot_path)
    logger.info("multi_mpc.plot_saved", path=str(plot_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RayMultiMPC over a planning model.")
    parser.add_argument("--vector",        type=str, default=None,
                        help="Override general.vector.")
    parser.add_argument("--ncpus",         type=int, default=None,
                        help="Override general.num_devices (parallel workers).")
    parser.add_argument("--n_sim",         type=int, default=None,
                        help="Total MPC runs. Defaults to general.num_instances from config.yml.")
    parser.add_argument(
        "--dynamic_price",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=None,
        help="Override general.dynamic_price. Pass True/False; omit to use config.yml.",
    )
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
    cfg = load_config(
        SCRIPTS_DIR / "config.yml",
        vector_override=args.vector,
        num_devices_override=args.ncpus,
    )
    # XLA flags / thread caps must be set before any jax import — and
    # RayMultiMPC's dispatch chain pulls jax via the worker imports.
    configure_jax_threads(cfg.nlp.n_devices, cfg.nlp.blas_threads)
    run_multisim(cfg, args)
