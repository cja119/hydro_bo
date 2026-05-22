"""Pareto-front study over the variance penalty `lambda`.

Background: the BO scoring rule is `g(x) = mu(x) - lam * sigma(x)`. Varying
`lam` traces a Pareto front in the (mean, stdev) plane — at `lam=0` BO
maximises mean alone; as `lam` grows BO trades mean for variance reduction.

Each PBS array index picks one lambda from a fixed list and runs the
constrained BO at 50% CI (`z_sc = 0`, `p_targ` from config) with the
full Sobol pool preloaded as the initial design. Only lambda varies
across cells. Output:

    scripts/tmp/lambda_pareto/<run-id>/<VECTOR>/lambda_NN_LLLL/
        args.json
        run.log
        bo_results_<ts>.csv
        bo_results_detailed_<ts>.json
        pareto_point.json     # the headline (lambda, mean, sd, score, x)

The headline `pareto_point.json` is what the post-hoc aggregator will
glob to assemble the curve once all tasks finish.

Default lambda grid is `[0, 1, 2, 5, 10, 20, 50, 100]` — span chosen
against the empirical (mean ≈ 2.17, sd ≈ 0.026) so the lower values
barely move the optimum and the upper values dominate the mean. Override
with `--lambdas 0,0.5,1,...` if you want a denser grid.

Two CLI sub-commands:

    python lambda_pareto_bo.py list-lambdas [--lambdas ...]
        Echoes the resolved lambda list (length = array size). Used by
        the shell submitter to derive `--array-size` without hard-coding.

    python lambda_pareto_bo.py run --array-index I --array-size N \
        --run-id RID [--lambdas L1,L2,...] [--iter-budget 8] \
        [--vector V] [--ncpus N]
        Single BO cell. PBS array entry point.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hydro_bo.mpc.dispatcher import RayMultiMPC, ensure_ray
from hydro_bo.utils.jax_threads import configure_jax_threads
from hydro_bo.utils.logging_config import configure_logging, get_logger
from hydro_bo.utils.run_config import (
    Config,
    load_config,
    merge_env_overrides,
    planning_model_path,
    resolve_sobol_dir,
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

# Pareto sweep grid. Picked against empirical (mean ≈ 2.17, sd ≈ 0.026)
# from a prior NH3 run: lam=1 docks the score by ~0.025 (negligible vs
# the ~2 mean spread), lam=100 docks it by ~2.6 (dominates). Upper end
# is a sanity check that the BO actually flips to variance-only.
_DEFAULT_LAMBDAS = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)

# At 50% CI the chance-bound constraint becomes a "median feasibility"
# constraint: `f(x) >= log_p_targ` (no σ buffer). z_sc = Φ⁻¹(0.5) = 0.
_Z_SC_50CI = 0.0

# Per-run mutable state. Module-level so the objective closure can write
# to the results log without threading another arg through.
_eval_counter: int = 0
_results_log: list = []
_bayesopt_dir: Optional[Path] = None
_run_timestamp: str = ""
_master_seed: int = 0


# ---------------------------------------------------------------------------
# CLI parsing helpers
# ---------------------------------------------------------------------------

def parse_lambda_list(raw: Optional[str]) -> tuple[float, ...]:
    """Resolve --lambdas; falls back to _DEFAULT_LAMBDAS. Accepts a
    comma-separated list of floats; ignores empty tokens so the user can
    pass leading/trailing commas without errors."""
    if not raw:
        return tuple(_DEFAULT_LAMBDAS)
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    if not tokens:
        return tuple(_DEFAULT_LAMBDAS)
    try:
        return tuple(float(t) for t in tokens)
    except ValueError as e:
        raise ValueError(
            f"--lambdas must be a comma-separated list of floats, got {raw!r}: {e}"
        )


def lambda_label(lam: float) -> str:
    """Filesystem-safe slug for the cell directory: `lambda_NN_LLLL`,
    where NN is the lambda's index in the sweep and LLLL is the value
    with the decimal point replaced by 'p' (`lambda_03_5p0`)."""
    return f"{lam:.3f}".rstrip("0").rstrip(".").replace(".", "p") or "0"


# ---------------------------------------------------------------------------
# Sobol cache loader (full pool, no row-list restriction)
# ---------------------------------------------------------------------------

def load_full_sobol_cache(
    sobol_dir: Path,
    cfg: Config,
    infeas_threshold: float,
    cap: Optional[int],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load every matching row under `sobol_dir` as (x, samples) for the
    BO's initial design. Mirrors the filter from
    `array_constrained_bo.load_sobol_cache`: infeasible / catastrophic
    entries are NaN'd to match what the live objective produces, and
    rows are silently skipped on (vector / bounds_expansion / dynamic_price)
    mismatch."""
    g = cfg.general
    observations: list[tuple[np.ndarray, np.ndarray]] = []
    n_missing = n_mismatched = 0

    row_dirs = sorted(sobol_dir.glob("row_*"))
    for rd in row_dirs:
        if cap is not None and len(observations) >= cap:
            break
        result_files = sorted(rd.glob("result_*.json"))
        if not result_files:
            n_missing += 1
            continue
        with open(result_files[-1]) as f:
            data = json.load(f)
        if (
            data.get("vector") != g.vector
            or float(data.get("bounds_expansion", -1)) != float(g.bounds_expansion)
            or bool(data.get("dynamic_price", False)) != bool(g.dynamic_price)
        ):
            n_mismatched += 1
            continue
        scores = data.get("worker_scores")
        if scores is None:
            obj = data.get("objective")
            scores = [obj] if obj is not None else []
        arr = np.asarray(scores, dtype=float).ravel()
        infeasible = ~np.isfinite(arr) | (arr <= infeas_threshold)
        arr = np.where(infeasible, np.nan, arr)
        observations.append((np.asarray(data["x"], dtype=float), arr))

    logger.info(
        "lambda_pareto.sobol_cache_loaded",
        dir=str(sobol_dir),
        n_rows_found=len(row_dirs),
        n_loaded=len(observations),
        n_missing_result=n_missing,
        n_mismatched=n_mismatched,
        cap=cap,
    )
    return observations


# ---------------------------------------------------------------------------
# Objective closure & results IO (same shape as ablation_bo / array_constrained_bo)
# ---------------------------------------------------------------------------

def _objective_factory(cfg: Config, ref: dict, infeas_threshold: float):
    g = cfg.general

    def objective(x: np.ndarray) -> np.ndarray:
        global _eval_counter
        _eval_counter += 1

        planning_params, env_overrides = params_from_x(
            x, ref, renewables=g.renewables, vector=g.vector
        )

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
        arr = (
            np.asarray(raw_scores, dtype=float).ravel()
            if raw_scores
            else np.array([], dtype=float)
        )
        if arr.size < g.num_instances:
            arr = np.concatenate([arr, np.full(g.num_instances - arr.size, np.nan)])
        infeasible = ~np.isfinite(arr) | (arr <= infeas_threshold)
        arr = np.where(infeasible, np.nan, arr)

        n_total = int(arr.size)
        k_feasible = int(n_total - int(infeasible.sum()))
        feasibility_rate = k_feasible / max(n_total, 1)
        mean_score = float(np.nanmean(arr)) if k_feasible >= 1 else float("nan")
        var_score = float(np.nanvar(arr, ddof=1)) if k_feasible >= 2 else float("nan")

        result_entry = {
            "eval_id": _eval_counter,
            "timestamp": datetime.now().isoformat(),
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
            "lambda_pareto.evaluation",
            eval_id=_eval_counter,
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


def save_results(results_log: list, bayesopt_dir: Path, run_timestamp: str):
    if not results_log:
        return
    fieldnames = [
        "eval_id", "timestamp", "mean_score", "var_score",
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


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------

def _cmd_list_lambdas(cli) -> None:
    """Print the resolved lambda list, one value per line. Lets the
    submitter shell derive `--array-size` without hard-coding the grid."""
    for lam in parse_lambda_list(cli.lambdas):
        print(lam)


def _cmd_run(cli) -> None:
    """Run a single Pareto cell. PBS array entry point."""
    cfg = load_config(
        SCRIPTS_DIR / "config.yml",
        vector_override=cli.vector,
        num_devices_override=cli.ncpus,
    )
    g, c = cfg.general, cfg.constrained_bo

    lambdas = parse_lambda_list(cli.lambdas)
    if cli.array_size != len(lambdas):
        raise ValueError(
            f"--array-size {cli.array_size} != len(lambdas) {len(lambdas)} "
            f"(resolved: {list(lambdas)}). The submitter must pass --array-size "
            f"matching the active --lambdas list."
        )
    if not 0 <= cli.array_index < len(lambdas):
        raise ValueError(
            f"--array-index {cli.array_index} out of range [0, {len(lambdas)})"
        )

    lam = float(lambdas[cli.array_index])
    label = f"lambda_{cli.array_index:02d}_{lambda_label(lam)}"

    # Start Ray BEFORE JAX threads spawn — see the dispatcher fork-deadlock
    # discussion in dispatcher.py / array_constrained_bo.py.
    ensure_ray(num_cpus=g.num_devices)
    configure_jax_threads(cfg.nlp.n_devices, cfg.nlp.blas_threads)
    # BO classes pull jax at import — defer until after configure_jax_threads.
    from hydro_bo.opt import ConstrainedBayesopt  # noqa: E402

    global _eval_counter, _results_log, _bayesopt_dir, _run_timestamp, _master_seed
    _eval_counter = 0
    _results_log = []

    now = datetime.now()
    _run_timestamp = now.strftime("%Y%m%d_%H%M%S")
    _bayesopt_dir = (
        SCRIPTS_DIR / "tmp" / "lambda_pareto" / cli.run_id / g.vector / label
    )
    _bayesopt_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(log_file=_bayesopt_dir / "run.log", package_level=g.log_level)

    _master_seed = resolve_master_seed(g.master_seed)
    iter_budget = int(cli.iter_budget)
    z_sc = _Z_SC_50CI

    logger.info(
        "lambda_pareto.cell",
        run_id=cli.run_id,
        array_index=cli.array_index,
        array_size=cli.array_size,
        lam=lam,
        z_sc=z_sc,
        p_targ=c.p_targ,
        iter_budget=iter_budget,
        label=label,
        master_seed=_master_seed,
    )

    with open(_bayesopt_dir / "args.json", "w") as f:
        json.dump(
            {
                "general": g.__dict__,
                "constrained_bo": c.__dict__,
                "nlp": cfg.nlp.__dict__,
                "lambda_pareto": {
                    "run_id": cli.run_id,
                    "array_index": cli.array_index,
                    "array_size": cli.array_size,
                    "lam": lam,
                    "label": label,
                    "iter_budget": iter_budget,
                    "z_sc": z_sc,
                    "p_targ": c.p_targ,
                    "lambdas": list(lambdas),
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
            "lambda_pareto.missing_planning_model",
            path=str(ref_path),
            message=f"Run: python scripts/planning.py {g.vector}",
        )
        return
    with open(ref_path) as f:
        ref = yaml.safe_load(f)

    bounds = build_bounds(ref, g.bounds_expansion)
    cat_vars = build_cat_vars(bounds)
    logger.info("lambda_pareto.search_space", dims=len(PARAM_KEYS))

    bo = ConstrainedBayesopt(
        f=_objective_factory(cfg, ref, c.infeas_threshold),
        bounds=bounds,
        n_initial_points=c.n_initial_points,
        iter_limit=iter_budget,
        lam=lam,                                # <-- the swept knob
        n_restarts=cfg.nlp.acq_n_restarts,
        pow_sobol=cfg.nlp.acq_pow_sobol,
        seed=_master_seed % (2**31),
        cat_vars=cat_vars,
        p_targ=c.p_targ,
        z_sc=z_sc,                              # <-- 50% CI = 0
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
    if sobol_dir is None:
        raise RuntimeError("constrained_bo.sobol_dir is unset in config.yml")
    preloaded = load_full_sobol_cache(
        sobol_dir, cfg, c.infeas_threshold, cap=c.n_sobol_cache,
    )
    for x, samples in preloaded:
        bo.observe(x, samples)
    if preloaded:
        # Skip BO's internal Sobol phase — we preloaded the entire design.
        bo.n_initial_points = len(preloaded)
        logger.info(
            "lambda_pareto.sobol_phase_skipped_via_cache",
            n_preloaded=len(preloaded),
        )

    best_x, best_score = bo.run()

    # Reconstruct (mean, sd) at the incumbent from the per-eval log. The
    # BO's `_best_observed` picks the empirically-feasible point with the
    # highest mu - lam·sd score, but it doesn't store the constituent
    # (mean, sd) — those live in `_results_log`. Match best_x against the
    # closest logged x to find them.
    best_x_arr = np.asarray(best_x, dtype=float).ravel()
    best_entry = None
    if _results_log:
        x_dists = [
            float(np.linalg.norm(np.asarray(e["x"], dtype=float) - best_x_arr))
            for e in _results_log
        ]
        best_entry = _results_log[int(np.argmin(x_dists))]
    best_mean = best_entry["mean_score"] if best_entry else float("nan")
    best_var = best_entry["var_score"] if best_entry else float("nan")
    best_sd = float(np.sqrt(best_var)) if np.isfinite(best_var) else float("nan")

    best_planning_params, best_env_overrides = params_from_x(
        best_x, ref, renewables=g.renewables, vector=g.vector
    )
    best_flat = flatten_dims(best_planning_params, best_env_overrides)

    pareto_point = {
        "run_id": cli.run_id,
        "array_index": cli.array_index,
        "lam": lam,
        "label": label,
        "z_sc": z_sc,
        "p_targ": c.p_targ,
        "iter_budget": iter_budget,
        "best_score": float(best_score),
        "best_mean": float(best_mean),
        "best_var": float(best_var),
        "best_sd": float(best_sd),
        "best_x": [float(v) for v in best_x_arr],
        "best_planning_params": {k: float(v) if isinstance(v, (int, float)) else v
                                 for k, v in best_planning_params.items()},
        "best_flat": {k: float(v) if isinstance(v, (int, float)) else v
                      for k, v in best_flat.items()},
    }
    with open(_bayesopt_dir / "pareto_point.json", "w") as f:
        json.dump(pareto_point, f, indent=2, default=str)

    logger.info(
        "lambda_pareto.complete",
        lam=lam,
        best_score=float(best_score),
        best_mean=float(best_mean),
        best_sd=float(best_sd),
    )

    save_results(_results_log, _bayesopt_dir, _run_timestamp)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=True)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser(
        "list-lambdas",
        help="Echo the resolved lambda list, one per line. Used by the "
             "shell submitter to set --array-size without hard-coding.",
    )
    p_list.add_argument("--lambdas", type=str, default=None,
                        help="Comma-separated list of lambdas; omit to use the "
                             "default sweep " + ",".join(str(v) for v in _DEFAULT_LAMBDAS))

    p_run = sub.add_parser(
        "run", help="Single Pareto cell. PBS array entry point.",
    )
    p_run.add_argument("--array-index", type=int, required=True,
                       help="0-based PBS_ARRAY_INDEX.")
    p_run.add_argument("--array-size", type=int, required=True,
                       help="Total cells in the array; must equal len(lambdas).")
    p_run.add_argument("--run-id", type=str, required=True,
                       help="Shared parent dir id (submission timestamp).")
    p_run.add_argument("--lambdas", type=str, default=None,
                       help="Comma-separated list of lambdas; omit to use the "
                            "default sweep " + ",".join(str(v) for v in _DEFAULT_LAMBDAS))
    p_run.add_argument("--iter-budget", type=int, default=8,
                       help="BO iterations *after* the full sobol pool is "
                            "preloaded as the initial design (default 8).")
    p_run.add_argument("--vector", type=str, default=None,
                       help="Hydrogen vector; overrides general.vector.")
    p_run.add_argument("--ncpus", type=int, default=None,
                       help="Override general.num_devices.")
    return parser


def main():
    parser = _build_parser()
    cli = parser.parse_args()
    if cli.cmd == "list-lambdas":
        _cmd_list_lambdas(cli)
    elif cli.cmd == "run":
        _cmd_run(cli)
    else:
        parser.error(f"unknown cmd {cli.cmd!r}")


if __name__ == "__main__":
    main()
