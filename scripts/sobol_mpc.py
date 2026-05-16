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


def _parse_cli_overrides() -> argparse.Namespace:
    """CLI overrides for config.yml. Two knobs only — vector and ncpus —
    everything else lives in scripts/config.yml. ncpus exists so PBS /
    shell wrappers can pass the queue's actual core count without editing
    the YAML; when omitted, `general.num_devices` from the YAML wins."""
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--vector", type=str, default=None,
                        help="Hydrogen vector for this run; overrides general.vector.")
    parser.add_argument("--ncpus", type=int, default=None,
                        help="Override general.num_devices (parallel workers).")
    return parser.parse_args()


def main():
    cli = _parse_cli_overrides()
    cfg = load_config(
        SCRIPTS_DIR / "config.yml",
        vector_override=cli.vector,
        num_devices_override=cli.ncpus,
    )
    g, s = cfg.general, cfg.sobol

    array_index = int(os.environ.get("PBS_ARRAY_INDEX", "0"))
    index_file = os.environ.get("SOBOL_INDEX_FILE")
    if index_file:
        # Topup mode: PBS_ARRAY_INDEX selects a line of `index_file`, and that
        # line's integer is the real Sobol row. Lets a sparse list of missing
        # rows be re-run via a contiguous PBS array.
        with open(index_file) as f:
            idx_list = [int(line.strip()) for line in f if line.strip()]
        index_row = idx_list[array_index]
    else:
        index_row = array_index

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

    # Streaming partial-results protocol:
    #   -inf  → slot not yet reported (still pending or never ran)
    #   null  → worker ran but failed
    #   finite → worker score
    worker_scores = [float("-inf")] * g.num_instances
    json_path = out_dir / f"result_{run_timestamp}.json"
    result = {
        "timestamp": now.isoformat(),
        "status": "running",
        "index_row": index_row,
        "vector": g.vector,
        "bounds_expansion": g.bounds_expansion,
        "dynamic_price": g.dynamic_price,
        "sobol_seed": s.seed,
        "sobol_pow_n": s.pow_n,
        "num_instances": g.num_instances,
        "num_devices": g.num_devices,
        "objective": None,
        "mean_score": None,
        "var_score": None,
        "n_workers": 0,
        "n_failed": 0,
        "worker_scores": worker_scores,
        "unit_sample": unit_sample.tolist(),
        "x": x.tolist(),
    }
    for key in PARAM_KEYS:
        result[key] = flat_dims[key]
    result["capex"] = planning_params["capex"]
    result["opex"] = planning_params["opex"]

    def _atomic_write_json(path, obj):
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=2)
        os.replace(tmp, path)

    def _recompute_stats():
        finite = np.array(
            [v for v in worker_scores if v is not None and np.isfinite(v)],
            dtype=float,
        )
        if finite.size > 0:
            mean_s = float(finite.mean())
            var_s = float(finite.var(ddof=1)) if finite.size >= 2 else float("nan")
            sd_s = float(np.sqrt(var_s)) if np.isfinite(var_s) else 0.0
            result["mean_score"] = mean_s
            result["var_score"] = var_s
            result["objective"] = mean_s - g.stdev_penalty * sd_s
        result["n_workers"] = sum(1 for v in worker_scores if v != float("-inf"))
        result["n_failed"] = sum(1 for v in worker_scores if v is None)

    _atomic_write_json(json_path, result)

    def _on_progress(worker_index, success, score, _relax_info):
        worker_scores[worker_index] = float(score) if success and score is not None else None
        _recompute_stats()
        try:
            _atomic_write_json(json_path, result)
        except Exception:
            logger.exception("sobol_mpc.partial_write_failed", worker_index=worker_index)

    try:
        dispatcher.run_multisim(deadline=deadline, progress_callback=_on_progress)
    except Exception:
        logger.exception("sobol_mpc.dispatcher_crashed", index_row=index_row)
        result["status"] = "dispatcher_crashed"
        _recompute_stats()
        _atomic_write_json(json_path, result)
        raise

    _recompute_stats()
    n_failed = result["n_failed"]
    n_not_run = sum(1 for v in worker_scores if v == float("-inf"))

    if result["objective"] is None:
        # No finite worker scores at all — fall back to the historical sentinel
        # so downstream filters (`obj <= -10.0`) treat the row as failed.
        result["objective"] = -1e8
        result["mean_score"] = float("nan")
        result["var_score"] = float("nan")

    if n_not_run > 0:
        result["status"] = "partial"
    elif n_failed > 0 and s.walltime_seconds is not None:
        result["status"] = "partial"
    else:
        result["status"] = "complete"

    logger.info(
        "sobol_mpc.complete",
        index_row=index_row,
        objective=result["objective"],
        mean_score=result["mean_score"],
        var_score=result["var_score"],
        n_failed=n_failed,
        n_not_run=n_not_run,
        n_workers=result["n_workers"],
    )

    _atomic_write_json(json_path, result)

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
