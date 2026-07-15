"""Array-job constrained Bayesian optimisation sweeping z_sc over a
linear range of one-sided confidence-interval levels.

Same as `constrained_bo.py`, but `z_sc` is derived from the task's
position in a PBS array:

    ci_pct_raw = array_index / (array_size - 1) * 100         # linspace 0..100
    ci_pct     = clip(ci_pct_raw, eps_pct, 100 - eps_pct)     # endpoints clipped
    z_sc       = Phi^{-1}(ci_pct / 100)                       # one-sided z-score

So for `--array-size 21`:
    index  0  -> ci  0.00% (clipped to  0.5%) -> z_sc ~ -2.576
    index 10  -> ci 50.00%                    -> z_sc =  0.000
    index 20  -> ci 100.00% (clipped to 99.5%) -> z_sc ~ +2.576

Outputs land in `scripts/tmp/array_cbo/<run-id>/<VECTOR>/idx_NN_ci_XX.X/`
so every task in one submission shares a parent. `--run-id` is set by
the PBS submitter once at submission time; if omitted, the script
falls back to the current wall-clock timestamp.

Everything else (iter_budget, n_initial_points, p_targ, warm-start
dirs, ...) still comes from `scripts/config.yml` — the only sweep
knob is `z_sc`.
"""

import argparse
import csv
import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from scipy.special import ndtri

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hydro_bo.mpc.dispatcher import RayMultiMPC
from hydro_bo.utils.jax_threads import configure_jax_threads
from hydro_bo.utils.logging_config import configure_logging, get_logger
from hydro_bo.utils.run_config import (
    Config,
    load_config,
    merge_env_overrides,
    planning_model_path,
    resolve_sobol_dir,
    resolve_warm_start_dirs,
)
from hydro_bo.utils.search_space import (
    PARAM_KEYS,
    build_bounds,
    build_cat_vars,
    flatten_dims,
    params_from_x,
)
from hydro_bo.utils.seeding import resolve_master_seed

logger = get_logger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parent

_DEFAULT_EPS_PCT = 0.5  # clip linspace endpoints to (0.5%, 99.5%)

_eval_counter: int = 0
_results_log: list = []
_bayesopt_dir: Optional[Path] = None
_run_timestamp: str = ""
_master_seed: int = 0


def _ci_for_task(array_index: int, array_size: int, eps_pct: float) -> tuple[float, float, float]:
    """Return (ci_pct_raw, ci_pct_clipped, z_sc) for this task.

    `array_size` of 1 is a special case: there's no linspace, so the
    sweep collapses to the midpoint (50%, z_sc=0)."""
    if array_size < 1:
        raise ValueError(f"array_size must be >= 1, got {array_size}")
    if not 0 <= array_index < array_size:
        raise ValueError(f"array_index {array_index} out of range [0, {array_size})")
    if array_size == 1:
        ci_raw = 50.0
    else:
        ci_raw = array_index / (array_size - 1) * 100.0
    ci_clipped = float(np.clip(ci_raw, eps_pct, 100.0 - eps_pct))
    z_sc = float(ndtri(ci_clipped / 100.0))
    return ci_raw, ci_clipped, z_sc


def _objective_factory(cfg: Config, ref: dict, infeas_threshold: float):
    """Build the BO `f` closure. Same shape as constrained_bo.py: runs
    N_INSTANCES MPC simulations and returns the per-worker sample array
    with non-finite / catastrophic entries marked NaN."""
    g = cfg.general

    def objective(x: np.ndarray) -> np.ndarray:
        global _eval_counter
        _eval_counter += 1

        planning_params, env_overrides = params_from_x(x, ref, renewables=g.renewables, vector=g.vector)

        dispatcher = RayMultiMPC(
            env_args=merge_env_overrides(cfg, env_overrides),
            param_overrides=planning_params,
            num_instances=g.num_instances,
            num_devices=g.num_devices,
            timeout=g.timeout,
            exit_fraction=1.0,
            master_seed=_master_seed + _eval_counter,
        )

        raw_scores = dispatcher.run_multisim()
        arr = np.asarray(raw_scores, dtype=float).ravel() if raw_scores else np.array([], dtype=float)
        if arr.size < g.num_instances:
            arr = np.concatenate([arr, np.full(g.num_instances - arr.size, np.nan)])
        infeasible = ~np.isfinite(arr) | (arr <= infeas_threshold)
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
            "x": [float(v) for v in np.asarray(x, dtype=float).ravel()],
        }
        flat_dims = flatten_dims(planning_params, env_overrides)
        for key in PARAM_KEYS:
            result_entry[key] = flat_dims[key]
        result_entry["capex"] = planning_params["capex"]
        result_entry["opex"] = planning_params["opex"]
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
    infeas_threshold: float,
    cap: Optional[int] = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load matching cached Sobol rows; identical filtering to
    constrained_bo.load_sobol_cache.

    `cap` is taken explicitly so callers can pass a value derived from a
    locally-replaced `constrained_bo` dataclass (e.g. after a CLI
    override). When omitted, falls back to `cfg.constrained_bo.n_sobol_cache`.
    """
    g = cfg.general
    if cap is None:
        cap = cfg.constrained_bo.n_sobol_cache
    observations: list[tuple[np.ndarray, np.ndarray]] = []
    n_missing_result = n_mismatched = 0

    row_dirs = sorted(sobol_dir.glob("row_*"))
    for rd in row_dirs:
        if cap is not None and len(observations) >= cap:
            break
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
        infeasible = ~np.isfinite(arr) | (arr <= infeas_threshold)
        arr = np.where(infeasible, np.nan, arr)
        observations.append((np.asarray(data["x"], dtype=float), arr))

    logger.info(
        "bayesopt.sobol_cache_loaded",
        dir=str(sobol_dir),
        n_rows_found=len(row_dirs),
        n_loaded=len(observations),
        n_missing_result=n_missing_result,
        n_mismatched=n_mismatched,
        cap=cap,
    )
    return observations


def load_bo_observations(
    warm_start_dirs: list,
    cfg: Config,
    infeas_threshold: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load observations from prior `constrained_bo` run directories
    (newest `bo_results_detailed_*.json` per dir)."""
    g = cfg.general
    observations: list[tuple[np.ndarray, np.ndarray]] = []
    n_missing = n_no_scores = 0

    for wsd in warm_start_dirs:
        if not wsd.exists():
            logger.warning("bayesopt.warm_start_dir_missing", path=str(wsd))
            n_missing += 1
            continue
        detailed_files = sorted(wsd.glob("bo_results_detailed_*.json"))
        if not detailed_files:
            logger.warning("bayesopt.warm_start_no_detailed", path=str(wsd))
            n_missing += 1
            continue
        with open(detailed_files[-1]) as f:
            entries = json.load(f)

        n_loaded_dir = 0
        for entry in entries:
            scores = entry.get("worker_scores")
            if not scores:
                n_no_scores += 1
                continue
            try:
                x = np.asarray([float(entry[k]) for k in PARAM_KEYS], dtype=float)
            except KeyError as missing:
                logger.warning(
                    "bayesopt.warm_start_missing_dim",
                    path=str(detailed_files[-1]),
                    eval_id=entry.get("eval_id"),
                    missing=str(missing),
                )
                continue
            arr = np.asarray(scores, dtype=float).ravel()
            if arr.size < g.num_instances:
                arr = np.concatenate([arr, np.full(g.num_instances - arr.size, np.nan)])
            infeasible = ~np.isfinite(arr) | (arr <= infeas_threshold)
            arr = np.where(infeasible, np.nan, arr)
            observations.append((x, arr))
            n_loaded_dir += 1

        logger.info(
            "bayesopt.warm_start_loaded",
            path=str(wsd),
            file=str(detailed_files[-1].name),
            n_entries=len(entries),
            n_loaded=n_loaded_dir,
        )

    logger.info(
        "bayesopt.warm_start_total",
        n_dirs=len(warm_start_dirs),
        n_missing=n_missing,
        n_no_scores=n_no_scores,
        n_loaded=len(observations),
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


def _parse_cli_overrides() -> argparse.Namespace:
    """CLI for the array sweep. `--array-index` and `--array-size` are
    required; everything else mirrors `constrained_bo.py`."""
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--array-index", type=int, required=True,
                        help="0-based index of this task in the CI sweep.")
    parser.add_argument("--array-size", type=int, required=True,
                        help="Total number of CI points in the sweep.")
    parser.add_argument("--eps-pct", type=float, default=_DEFAULT_EPS_PCT,
                        help=f"Endpoint clip in percent (default {_DEFAULT_EPS_PCT}). "
                             "ci is clipped to [eps, 100-eps] before Φ⁻¹.")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Shared parent directory id for the whole array. "
                             "If omitted, falls back to the current timestamp.")
    parser.add_argument("--vector", type=str, default=None,
                        help="Hydrogen vector for this run; overrides general.vector.")
    parser.add_argument("--ncpus", type=int, default=None,
                        help="Override general.num_devices (parallel workers).")
    parser.add_argument("--n-sobol-cache", type=int, default=None,
                        help="Override constrained_bo.n_sobol_cache (cap on cached "
                             "Sobol rows preloaded into the BO). Omit to use the "
                             "value from config.yml.")
    return parser.parse_args()


def main():
    cli = _parse_cli_overrides()
    cfg = load_config(
        SCRIPTS_DIR / "config.yml",
        vector_override=cli.vector,
        num_devices_override=cli.ncpus,
    )
    g, c, s = cfg.general, cfg.constrained_bo, cfg.sobol

    ci_raw, ci_clipped, z_sc = _ci_for_task(cli.array_index, cli.array_size, cli.eps_pct)
    c = replace(c, z_sc=z_sc)
    if cli.n_sobol_cache is not None:
        c = replace(c, n_sobol_cache=int(cli.n_sobol_cache))

    configure_jax_threads(cfg.nlp.n_devices, cfg.nlp.blas_threads)
    from hydro_bo.mpc.dispatcher import ensure_ray  # noqa: E402
    ensure_ray(num_cpus=cfg.general.num_devices)
    from hydro_bo.opt import ConstrainedBayesopt  # noqa: E402

    global _eval_counter, _results_log, _bayesopt_dir, _run_timestamp, _master_seed
    _eval_counter = 0
    _results_log = []

    now = datetime.now()
    _run_timestamp = now.strftime("%Y%m%d_%H%M%S")
    run_id = cli.run_id or _run_timestamp
    task_subdir = f"idx_{cli.array_index:03d}_of_{cli.array_size:03d}_ci_{ci_raw:05.1f}"
    _bayesopt_dir = SCRIPTS_DIR / "tmp" / "array_cbo" / run_id / g.vector / task_subdir
    _bayesopt_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(log_file=_bayesopt_dir / "run.log", package_level=g.log_level)

    _master_seed = resolve_master_seed(g.master_seed)
    logger.info("bayesopt.master_seed", master_seed=_master_seed, cli_seed=g.master_seed)
    logger.info(
        "bayesopt.array_sweep",
        array_index=cli.array_index,
        array_size=cli.array_size,
        ci_pct_raw=ci_raw,
        ci_pct_clipped=ci_clipped,
        eps_pct=cli.eps_pct,
        z_sc=z_sc,
        run_id=run_id,
    )
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
                "nlp": cfg.nlp.__dict__,
                "array": {
                    "index": cli.array_index,
                    "size": cli.array_size,
                    "eps_pct": cli.eps_pct,
                    "ci_pct_raw": ci_raw,
                    "ci_pct_clipped": ci_clipped,
                    "z_sc": z_sc,
                    "run_id": run_id,
                },
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
        f=_objective_factory(cfg, ref, c.infeas_threshold),
        bounds=bounds,
        n_initial_points=c.n_initial_points,
        iter_limit=c.iter_budget,
        lam=g.stdev_penalty,
        n_restarts=cfg.nlp.acq_n_restarts,
        pow_sobol=cfg.nlp.acq_pow_sobol,
        seed=_master_seed % (2**31),
        cat_vars=cat_vars,
        p_targ=c.p_targ,
        z_sc=c.z_sc,
        l1_penalty=c.l1_penalty,
        sqp_config=cfg.nlp.to_sqp_config(),
        pad_initial=cfg.nlp.pad_initial,
        gp_lbfgs_max_iter=cfg.nlp.gp_lbfgs_max_iter,
        gp_mu_kernel=cfg.nlp.gp_mu_kernel,
        gp_log_var_kernel=cfg.nlp.gp_log_var_kernel,
        gp_bin_kernel=cfg.nlp.gp_bin_kernel,
        gp_bin_label_smoothing=cfg.nlp.gp_bin_label_smoothing,
    )

    sobol_dir = resolve_sobol_dir(c.sobol_dir, SCRIPTS_DIR, g.vector)
    n_preloaded = 0
    if sobol_dir is not None:
        if not sobol_dir.exists():
            logger.error("bayesopt.sobol_dir_missing", path=str(sobol_dir))
        else:
            preloaded = load_sobol_cache(
                sobol_dir, cfg, c.infeas_threshold, cap=c.n_sobol_cache,
            )
            for x, samples in preloaded:
                bo.observe(x, samples)
            n_preloaded += len(preloaded)

    warm_start_dirs = resolve_warm_start_dirs(c.warm_start_dirs, SCRIPTS_DIR, g.vector)
    if warm_start_dirs:
        warm = load_bo_observations(warm_start_dirs, cfg, c.infeas_threshold)
        for x, samples in warm:
            bo.observe(x, samples)
        n_preloaded += len(warm)

    if n_preloaded:
        bo.n_initial_points = n_preloaded
        logger.info("bayesopt.sobol_phase_skipped_via_cache", n_preloaded=n_preloaded)

    best_x, best_score = bo.run()
    best_planning_params, best_env_overrides = params_from_x(
        best_x, ref, renewables=g.renewables, vector=g.vector
    )
    best_flat = flatten_dims(best_planning_params, best_env_overrides)

    logger.info("bayesopt.complete")
    logger.info("bayesopt.best_score", score=best_score)
    for k, v in best_flat.items():
        logger.info("bayesopt.parameter", name=k, value=v)

    out_path = _bayesopt_dir / f"{g.vector}-Chile-cbo.yml"
    snapshot = dict(best_planning_params)
    snapshot["wind_forecast_mean"] = best_flat["wind_forecast_mean"]
    with open(out_path, "w") as f:
        yaml.dump(snapshot, f, default_flow_style=False)
    logger.info("bayesopt.saved", path=str(out_path))

    save_results(_results_log, _bayesopt_dir, _run_timestamp)
    return snapshot, best_score


if __name__ == "__main__":
    main()
