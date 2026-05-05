"""Run a single Sobol-sampled MPC evaluation.

Reads `scripts/config.yml` for everything; takes the Sobol row index
from the `PBS_ARRAY_INDEX` env var (default 0). The Sobol sequence is
regenerated deterministically from `cfg.sobol.seed` each invocation —
no shared `sobol_indices.csv` to keep aligned with downstream consumers.
"""

import argparse
import csv
import json
import os
import sys
import time as _time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hydro_bo.mpc.dispatcher import RayMultiMPC
from hydro_bo.utils.logging_config import configure_logging, get_logger
from hydro_bo.utils.run_config import (
    load_config,
    merge_env_overrides,
    planning_model_path,
    resolve_sobol_dir,
)
from hydro_bo.utils.search_space import (
    PARAM_KEYS,
    build_bounds,
    flatten_dims,
    params_from_x,
    scale_unit_to_bounds,
    sobol_unit_sample,
)
from hydro_bo.utils.seeding import resolve_master_seed

logger = get_logger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parent


def _parse_vector_arg() -> str | None:
    """Single-arg CLI: `--vector` is the only flag that overrides config.
    Everything else lives in scripts/config.yml."""
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--vector", type=str, default=None,
                        help="Hydrogen vector for this run; overrides general.vector.")
    return parser.parse_args().vector


def main():
    vector = _parse_vector_arg()
    cfg = load_config(SCRIPTS_DIR / "config.yml", vector_override=vector)
    g, s = cfg.general, cfg.sobol

    index_row = int(os.environ.get("PBS_ARRAY_INDEX", "0"))

    now = datetime.now()
    run_timestamp = now.strftime("%Y%m%d_%H%M%S")
    sobol_base = resolve_sobol_dir(s.sobol_dir, SCRIPTS_DIR, g.vector)
    if sobol_base is None:
        # Back-compat fallback if `sobol.sobol_dir` is unset in the YAML.
        run_label = f"{g.vector}-dynamic_price" if g.dynamic_price else g.vector
        sobol_base = SCRIPTS_DIR / "tmp" / "sobol" / run_label
    out_dir = sobol_base / f"row_{index_row:05d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(log_file=out_dir / "run.log", package_level=g.log_level)

    master_seed = resolve_master_seed(g.master_seed)
    logger.info(
        "sobol_mpc.master_seed",
        master_seed=master_seed,
        cli_seed=g.master_seed,
        index_row=index_row,
    )

    with open(out_dir / "args.json", "w") as f:
        json.dump(
            {
                "general": g.__dict__,
                "sobol": s.__dict__,
                "index_row": index_row,
                "master_seed_resolved": master_seed,
            },
            f,
            indent=2,
            default=str,
        )

    ref_path = planning_model_path(SCRIPTS_DIR, g.vector)
    if not ref_path.exists():
        logger.error(
            "sobol_mpc.missing_planning_model",
            path=str(ref_path),
            message=f"Run: python scripts/planning.py {g.vector}",
        )
        return
    with open(ref_path) as f:
        ref = yaml.safe_load(f)

    bounds = build_bounds(ref, g.bounds_expansion)
    unit_sample = sobol_unit_sample(seed=s.seed, pow_n=s.pow_n, index_row=index_row)
    x = scale_unit_to_bounds(unit_sample, bounds)

    logger.info("sobol_mpc.bounds_built", bounds_expansion=g.bounds_expansion)
    logger.info(
        "sobol_mpc.sample_loaded",
        index_row=index_row,
        sobol_seed=s.seed,
        sobol_pow_n=s.pow_n,
        unit_sample=unit_sample.tolist(),
        x=x.tolist(),
    )

    planning_params, env_overrides = params_from_x(x, ref, renewables=g.renewables, vector=g.vector)
    flat_dims = flatten_dims(planning_params, env_overrides)
    logger.info("sobol_mpc.parameters", **flat_dims)

    deadline = None
    if s.walltime_seconds is not None:
        deadline = _time.perf_counter() + s.walltime_seconds - s.buffer_seconds
        logger.info(
            "sobol_mpc.deadline_set",
            walltime_seconds=s.walltime_seconds,
            buffer_seconds=s.buffer_seconds,
        )

    dispatcher = RayMultiMPC(
        env_args=merge_env_overrides(cfg, env_overrides),
        param_overrides=planning_params,
        num_instances=g.num_instances,
        num_devices=g.num_devices,
        timeout=g.timeout,
        exit_fraction=1.0,
        master_seed=master_seed,
    )
    logger.info(
        "sobol_mpc.dispatcher_initialized",
        num_instances=g.num_instances,
        num_devices=g.num_devices,
        timeout=g.timeout,
    )

    scores = dispatcher.run_multisim(deadline=deadline)
    scores_arr = np.array(scores, dtype=float) if scores else np.array([], dtype=float)
    finite_mask = np.isfinite(scores_arr)
    finite_scores = scores_arr[finite_mask]
    n_failed = int((~finite_mask).sum())

    if len(finite_scores) > 0:
        mean_score = float(finite_scores.mean())
        var_score = float(finite_scores.var(ddof=1)) if len(finite_scores) >= 2 else float("nan")
        sd_score = float(np.sqrt(var_score)) if np.isfinite(var_score) else 0.0
        objective = mean_score - g.stdev_penalty * sd_score
    else:
        mean_score = float("nan")
        var_score = float("nan")
        objective = -1e8

    logger.info(
        "sobol_mpc.complete",
        index_row=index_row,
        objective=objective,
        mean_score=mean_score,
        var_score=var_score,
        n_failed=n_failed,
        n_workers=len(scores),
    )

    status = "complete" if s.walltime_seconds is None or n_failed == 0 else "partial"
    result = {
        "timestamp": now.isoformat(),
        "status": status,
        "index_row": index_row,
        "vector": g.vector,
        "bounds_expansion": g.bounds_expansion,
        "dynamic_price": g.dynamic_price,
        "sobol_seed": s.seed,
        "sobol_pow_n": s.pow_n,
        "num_instances": g.num_instances,
        "num_devices": g.num_devices,
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
        result[key] = flat_dims[key]
    result["capex"] = planning_params["capex"]
    result["opex"] = planning_params["opex"]

    json_path = out_dir / f"result_{run_timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    csv_fieldnames = (
        [
            "timestamp", "index_row", "vector", "bounds_expansion", "dynamic_price",
            "sobol_seed", "sobol_pow_n", "num_instances", "num_devices",
            "objective", "mean_score", "var_score", "n_workers",
        ]
        + PARAM_KEYS
        + ["capex", "opex"]
    )
    csv_path = out_dir / f"result_{run_timestamp}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(result)
    logger.info("sobol_mpc.result_saved", path=str(json_path))


if __name__ == "__main__":
    main()
