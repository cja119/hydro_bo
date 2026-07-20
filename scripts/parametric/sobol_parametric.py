"""Sobol sampling over the JOINT space [x_design | theta].

One PBS array task evaluates one Sobol row: `PBS_ARRAY_INDEX` selects the
row, the row is decoded into planning params + theta sinks, and the MPC
multisim is dispatched. Results land in

    <sobol_dir>/row_NNNNN/result_<timestamp>.json

in the same schema `scripts/general/sobol_mpc.py` writes, so the KG BO
can replay them as initial observations — except that `x` here is the
*joint* vector and the file carries a `theta_params` list, which the
loader checks so a cache built for a different theta set is rejected
rather than silently misread.

The Sobol sequence is regenerated deterministically from
(seed, pow_n, index_row), so array tasks need no shared state.

Reads `scripts/parametric/config.yml`.
"""

import argparse
import json
import os
import sys
import time as _time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
from hydro_bo.utils.theta import registry_from_names

logger = get_logger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parent


def _deep_merge(target: dict, updates: dict) -> None:
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(target.get(k), dict):
            _deep_merge(target[k], v)
        else:
            target[k] = v


def _atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _parse_cli_overrides() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--vector", type=str, default=None,
                        help="Hydrogen vector; overrides general.vector.")
    parser.add_argument("--ncpus", type=int, default=None,
                        help="Override general.num_devices (parallel workers).")
    parser.add_argument("--index-row", type=int, default=None,
                        help="Sobol row; defaults to PBS_ARRAY_INDEX, then 0.")
    parser.add_argument("--index-file", type=str, default=None,
                        help="Topup mode: file of row indices, one per line; "
                             "PBS_ARRAY_INDEX selects a line.")
    return parser.parse_args()


def main():
    cli = _parse_cli_overrides()
    cfg = load_config(
        SCRIPTS_DIR / "config.yml",
        vector_override=cli.vector,
        num_devices_override=cli.ncpus,
    )
    g, s = cfg.general, cfg.sobol
    if cfg.theta is None:
        raise SystemExit("config.yml has no `theta:` block — required here.")

    registry = registry_from_names(cfg.theta.params, vector=g.vector)
    d_design, d_theta = len(PARAM_KEYS), registry.dim
    d_total = d_design + d_theta

    array_index = int(os.environ.get("PBS_ARRAY_INDEX", "0"))
    if cli.index_row is not None:
        index_row = cli.index_row
    elif cli.index_file:
        # Topup: PBS_ARRAY_INDEX selects a line of the index file, so a
        # sparse set of missing rows can be re-run as a contiguous array.
        idx_list = [
            int(v) for v in Path(cli.index_file).read_text().split() if v.strip()
        ]
        index_row = idx_list[array_index]
    else:
        index_row = array_index

    sobol_base = resolve_sobol_dir(s.sobol_dir, SCRIPTS_DIR, g.vector)
    if sobol_base is None:
        raise SystemExit("sobol.sobol_dir must be set in config.yml")
    out_dir = sobol_base / f"row_{index_row:05d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    run_timestamp = now.strftime("%Y%m%d_%H%M%S")
    configure_logging(log_file=out_dir / "run.log", package_level=g.log_level)

    master_seed = resolve_master_seed(g.master_seed)
    logger.info(
        "sobol_parametric.start",
        index_row=index_row, vector=g.vector,
        d_design=d_design, d_theta=d_theta, master_seed=master_seed,
        theta_params=registry.names,
    )

    ref_path = planning_model_path(SCRIPTS_DIR, g.vector, g.planning_dir)
    if not ref_path.exists():
        raise SystemExit(
            f"planning model missing: {ref_path}\n"
            f"Run: python scripts/general/planning_model.py {g.vector}"
        )
    ref = yaml.safe_load(ref_path.read_text())

    bounds = np.vstack([build_bounds(ref, g.bounds_expansion), registry.bounds()])
    unit = sobol_unit_sample(
        seed=s.seed, pow_n=s.pow_n, index_row=index_row, dim=d_total,
    )
    x_joint = scale_unit_to_bounds(unit, bounds)
    x_design, theta = x_joint[:d_design], x_joint[d_design:]

    bundle = registry.apply(theta)
    planning_params, env_overrides = params_from_x(
        x_design, ref, renewables=g.renewables, vector=g.vector,
        cost_overrides=bundle.cost_overrides,
    )
    if bundle.env_overrides:
        _deep_merge(env_overrides, bundle.env_overrides)
    flat_dims = flatten_dims(planning_params, env_overrides)

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
        "theta_params": registry.names,
        "objective": None,
        "mean_score": None,
        "var_score": None,
        "n_workers": 0,
        "n_failed": 0,
        "worker_scores": [],
        "unit_sample": unit.tolist(),
        "x": x_joint.tolist(),          # JOINT vector — what the BO observes
        "x_design": x_design.tolist(),
        "theta": theta.tolist(),
    }
    for key in PARAM_KEYS:
        result[key] = flat_dims[key]
    for name, val in zip(registry.names, theta):
        result[f"theta.{name}"] = float(val)
    result["capex"] = planning_params["capex"]
    result["opex"] = planning_params["opex"]
    _atomic_write_json(json_path, result)

    for key, (lo, hi) in zip(PARAM_KEYS + registry.names, bounds):
        logger.info("sobol_parametric.bounds", parameter=key, lower=lo, upper=hi)

    if s.walltime_seconds is not None:
        budget = s.walltime_seconds - s.buffer_seconds
        logger.info("sobol_parametric.deadline", seconds_available=budget)

    t0 = _time.perf_counter()
    try:
        dispatcher = RayMultiMPC(
            env_args=merge_env_overrides(cfg, env_overrides),
            param_overrides=planning_params,
            num_instances=g.num_instances,
            num_devices=g.num_devices,
            timeout=g.timeout,
            exit_fraction=1.0,
            master_seed=master_seed,
        )
        raw_scores = dispatcher.run_multisim()
    except Exception:
        logger.exception("sobol_parametric.dispatcher_crashed", index_row=index_row)
        result["status"] = "crashed"
        _atomic_write_json(json_path, result)
        raise
    elapsed = _time.perf_counter() - t0

    arr = (
        np.asarray(raw_scores, dtype=float).ravel()
        if raw_scores else np.array([], dtype=float)
    )
    if arr.size < g.num_instances:
        arr = np.concatenate([arr, np.full(g.num_instances - arr.size, np.nan)])
    finite = arr[np.isfinite(arr)]

    result["status"] = "complete"
    result["worker_scores"] = [
        None if not np.isfinite(v) else float(v) for v in arr
    ]
    result["n_workers"] = int(len(raw_scores))
    result["n_failed"] = int(arr.size - finite.size)
    result["mean_score"] = float(finite.mean()) if finite.size else None
    result["var_score"] = (
        float(finite.var(ddof=1)) if finite.size >= 2 else None
    )
    if finite.size:
        sd = float(np.sqrt(result["var_score"])) if result["var_score"] else 0.0
        result["objective"] = float(finite.mean()) - g.stdev_penalty * sd
    result["elapsed_seconds"] = elapsed
    _atomic_write_json(json_path, result)

    logger.info(
        "sobol_parametric.complete",
        index_row=index_row,
        objective=result["objective"],
        n_workers=result["n_workers"],
        n_failed=result["n_failed"],
        elapsed_seconds=round(elapsed, 1),
        path=str(json_path),
    )
    return result


if __name__ == "__main__":
    main()
