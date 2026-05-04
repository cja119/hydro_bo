"""Constrained Bayesian optimisation over the MPC planning model parameters.

Same shape as `unconstrained_bo.py` but pairs the noisy-EI objective
with a Polya-Gamma BinomialGP feasibility surrogate and a chance-bound
constraint enforced inside the SQP. Failed solves are preserved as NaN
samples (rather than substituted with a penalty value) so the
BinomialGP picks up the feasibility signal directly.

Reads `scripts/config.yml` for everything — no argparse, no CAPS-block.
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hydro_bo.mpc.dispatcher import RayMultiMPC
from hydro_bo.utils.jax_threads import configure_jax_threads
from hydro_bo.utils.logging_config import configure_logging, get_logger
from hydro_bo.utils.run_config import (
    Config,
    env_args_from,
    load_config,
    planning_model_path,
    resolve_sobol_dir,
)
from hydro_bo.utils.search_space import (
    INTEGER_KEYS,
    PARAM_KEYS,
    build_bounds,
    build_cat_vars,
    params_from_x,
)
from hydro_bo.utils.seeding import resolve_master_seed

logger = get_logger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parent

_eval_counter: int = 0
_results_log: list = []
_bayesopt_dir: Optional[Path] = None
_run_timestamp: str = ""
_master_seed: int = 0


def _objective_factory(cfg: Config, ref: dict):
    """Build the BO `f` closure that runs N_INSTANCES MPC simulations
    and returns the per-worker sample array with non-finite /
    catastrophic entries marked NaN. The constrained BO derives k
    (feasible) and N (total) directly from these arrays."""
    g = cfg.general
    c = cfg.constrained_bo

    def objective(x: np.ndarray) -> np.ndarray:
        global _eval_counter
        _eval_counter += 1

        params = params_from_x(x, ref, renewables=g.renewables, vector=g.vector)

        dispatcher = RayMultiMPC(
            env_args=env_args_from(cfg),
            param_overrides=params,
            num_instances=g.num_instances,
            num_devices=g.num_devices,
            timeout=g.timeout,
            exit_fraction=1.0,
            master_seed=_master_seed + _eval_counter,
        )

        raw_scores = dispatcher.run_multisim()
        arr = np.asarray(raw_scores, dtype=float).ravel() if raw_scores else np.array([], dtype=float)
        # Pad missing workers with NaN — a worker that never reported back
        # is still a feasibility signal at this x (timeout / OOM).
        if arr.size < g.num_instances:
            arr = np.concatenate([arr, np.full(g.num_instances - arr.size, np.nan)])
        infeasible = ~np.isfinite(arr) | (arr <= c.infeas_threshold)
        arr = np.where(infeasible, np.nan, arr)

        n_total = int(arr.size)
        k_feasible = int(n_total - int(infeasible.sum()))
        feasibility_rate = k_feasible / max(n_total, 1)
        mean_score = float(np.nanmean(arr)) if k_feasible >= 1 else float("nan")
        var_score = float(np.nanvar(arr, ddof=1)) if k_feasible >= 2 else float("nan")
        sd_score = float(np.sqrt(var_score)) if np.isfinite(var_score) else 0.0
        objective_value = (
            mean_score - g.stdev_penalty * sd_score
            if np.isfinite(mean_score)
            else float("nan")
        )

        result_entry = {
            "eval_id": _eval_counter,
            "timestamp": datetime.now().isoformat(),
            "objective": objective_value,
            "mean_score": mean_score,
            "var_score": var_score,
            "num_workers": len(raw_scores),
            "k_feasible": k_feasible,
            "n_total": n_total,
            "feasibility_rate": feasibility_rate,
            "worker_scores": list(raw_scores),
        }
        for key in PARAM_KEYS:
            result_entry[key] = params[key]
        result_entry["capex"] = params["capex"]
        result_entry["opex"] = params["opex"]
        _results_log.append(result_entry)

        logger.info(
            "bayesopt.evaluation",
            eval_id=_eval_counter,
            objective=objective_value,
            mean=mean_score,
            variance=var_score,
            k_feasible=k_feasible,
            n_total=n_total,
            feasibility_rate=feasibility_rate,
        )
        if _bayesopt_dir is not None:
            save_results(_results_log, _bayesopt_dir, _run_timestamp)
        return arr

    return objective


def load_sobol_cache(
    sobol_dir: Path,
    cfg: Config,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load every matching cached row from a sobol_mpc results
    directory. Non-finite / catastrophic worker scores become NaN, and
    rows are NaN-padded to `num_instances` so `N` is consistent."""
    g, c = cfg.general, cfg.constrained_bo
    observations: list[tuple[np.ndarray, np.ndarray]] = []
    n_missing_result = n_mismatched = 0

    row_dirs = sorted(sobol_dir.glob("row_*"))
    for rd in row_dirs:
        result_files = sorted(rd.glob("result_*.json"))
        if not result_files:
            n_missing_result += 1
            continue
        with open(result_files[-1]) as f:
            data = json.load(f)

        if data.get("vector") != g.vector \
                or float(data.get("bounds_expansion", -1)) != float(g.bounds_expansion) \
                or bool(data.get("dynamic_price", False)) != bool(g.dynamic_price):
            n_mismatched += 1
            continue

        scores = data.get("worker_scores")
        if scores is None:
            obj = data.get("objective")
            scores = [obj] if obj is not None else []
        arr = np.asarray(scores, dtype=float).ravel() if scores else np.array([], dtype=float)
        if arr.size < g.num_instances:
            arr = np.concatenate([arr, np.full(g.num_instances - arr.size, np.nan)])
        infeasible = ~np.isfinite(arr) | (arr <= c.infeas_threshold)
        arr = np.where(infeasible, np.nan, arr)
        observations.append((np.asarray(data["x"], dtype=float), arr))

    logger.info(
        "bayesopt.sobol_cache_loaded",
        dir=str(sobol_dir),
        n_rows_found=len(row_dirs),
        n_loaded=len(observations),
        n_missing_result=n_missing_result,
        n_mismatched=n_mismatched,
    )
    return observations


def save_results(results_log: list, bayesopt_dir: Path, run_timestamp: str):
    if not results_log:
        return
    fieldnames = [
        "eval_id", "timestamp", "objective", "mean_score", "var_score",
        "num_workers", "k_feasible", "n_total", "feasibility_rate",
    ] + PARAM_KEYS + ["capex", "opex"]
    csv_path = bayesopt_dir / f"bo_results_{run_timestamp}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for entry in results_log:
            writer.writerow({k: v for k, v in entry.items() if k != "worker_scores"})
    json_path = bayesopt_dir / f"bo_results_detailed_{run_timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(results_log, f, indent=2)


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
    g, c, s = cfg.general, cfg.constrained_bo, cfg.sobol

    configure_jax_threads(g.num_devices)
    from hydro_bo.opt import ConstrainedBayesopt  # noqa: E402

    global _eval_counter, _results_log, _bayesopt_dir, _run_timestamp, _master_seed
    _eval_counter = 0
    _results_log = []

    now = datetime.now()
    _run_timestamp = now.strftime("%Y%m%d_%H%M%S")
    _bayesopt_dir = (
        SCRIPTS_DIR / "tmp" / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S") / g.vector
    )
    _bayesopt_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(log_file=_bayesopt_dir / "run.log", package_level=g.log_level)

    _master_seed = resolve_master_seed(g.master_seed)
    logger.info("bayesopt.master_seed", master_seed=_master_seed, cli_seed=g.master_seed)
    logger.info(
        "bayesopt.constrained_config",
        p_targ=c.p_targ,
        z_sc=c.z_sc,
        l1_penalty=c.l1_penalty,
        infeas_threshold=c.infeas_threshold,
    )

    with open(_bayesopt_dir / "args.json", "w") as f:
        json.dump(
            {
                "general": g.__dict__,
                "constrained_bo": c.__dict__,
                "sobol": s.__dict__,
                "master_seed_resolved": _master_seed,
            },
            f,
            indent=2,
            default=str,
        )

    ref_path = planning_model_path(SCRIPTS_DIR, g.vector)
    if not ref_path.exists():
        logger.error(
            "bayesopt.missing_planning_model",
            path=str(ref_path),
            message=f"Run: python scripts/planning.py {g.vector}",
        )
        return None, None
    with open(ref_path) as f:
        ref = yaml.safe_load(f)

    bounds = build_bounds(ref, g.bounds_expansion)
    cat_vars = build_cat_vars(bounds)
    logger.info("bayesopt.search_space", dims=len(PARAM_KEYS))
    for key, (lo, hi) in zip(PARAM_KEYS, bounds):
        logger.info("bayesopt.parameter_bounds", parameter=key, lower=lo, upper=hi)

    bo = ConstrainedBayesopt(
        f=_objective_factory(cfg, ref),
        bounds=bounds,
        n_initial_points=c.n_initial_points,
        iter_limit=c.iter_budget,
        lam=g.stdev_penalty,
        n_restarts=5,
        seed=_master_seed % (2**31),
        cat_vars=cat_vars,
        p_targ=c.p_targ,
        z_sc=c.z_sc,
        l1_penalty=c.l1_penalty,
    )

    sobol_dir = resolve_sobol_dir(c.sobol_dir, SCRIPTS_DIR, g.vector)
    if sobol_dir is not None:
        if not sobol_dir.exists():
            logger.error("bayesopt.sobol_dir_missing", path=str(sobol_dir))
        else:
            preloaded = load_sobol_cache(sobol_dir, cfg)
            for x, samples in preloaded:
                bo.observe(x, samples)
            if preloaded:
                bo.n_initial_points = len(preloaded)
                logger.info("bayesopt.sobol_phase_skipped_via_cache", n_preloaded=len(preloaded))

    best_x, best_score = bo.run()
    best_params = params_from_x(best_x, ref, renewables=g.renewables, vector=g.vector)

    logger.info("bayesopt.complete")
    logger.info("bayesopt.best_score", score=best_score)
    for k, v in best_params.items():
        logger.info("bayesopt.parameter", name=k, value=v)

    out_path = SCRIPTS_DIR / "tmp" / "planning" / f"{g.vector}-Chile-cbo.yml"
    with open(out_path, "w") as f:
        yaml.dump(best_params, f, default_flow_style=False)
    logger.info("bayesopt.saved", path=str(out_path))

    save_results(_results_log, _bayesopt_dir, _run_timestamp)
    return best_params, best_score


if __name__ == "__main__":
    main()
