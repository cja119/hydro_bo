"""Unconstrained parametric BO driven by the knowledge gradient.

The GP is fit over the joint space [x_design | theta], and the
acquisition is

    KG(x,theta) = E_z[ max_x' E_theta'[ mu_{n+1}(x',theta' | z,x,theta) ] ]

so the outer optimisation returns (x*, theta*) jointly — theta* being the
contextual query expected to be most informative, rather than a value
drawn from a schedule.

Feasibility / chance constraints are deliberately not applied at this
stage: this is the end-to-end validation run for the KG pipeline.

Reads `scripts/parametric/config.yml` (`kg:` block).
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
)
from hydro_bo.utils.search_space import (
    PARAM_KEYS,
    build_bounds,
    build_cat_vars,
    flatten_dims,
    params_from_x,
)
from hydro_bo.utils.seeding import resolve_master_seed
from hydro_bo.utils.theta import registry_from_names

logger = get_logger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parent

_eval_counter: int = 0
_results_log: list = []
_bayesopt_dir: Optional[Path] = None
_run_timestamp: str = ""
_master_seed: int = 0


def _deep_merge(target: dict, updates: dict) -> None:
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(target.get(k), dict):
            _deep_merge(target[k], v)
        else:
            target[k] = v


def _objective_factory(cfg: Config, ref: dict, registry):
    """BO `f` closure over the joint vector `[x_design | theta]`. Theta is
    split off and routed through the registry's cost / env / MPC sinks;
    the design half decodes exactly as it does for an IDC run."""
    g = cfg.general
    c = cfg.constrained_bo
    d_design = len(PARAM_KEYS)

    def objective(x_joint: np.ndarray) -> np.ndarray:
        global _eval_counter
        _eval_counter += 1

        x_joint = np.asarray(x_joint, dtype=float).ravel()
        x, theta = x_joint[:d_design], x_joint[d_design:]
        bundle = registry.apply(theta)

        planning_params, env_overrides = params_from_x(
            x, ref, renewables=g.renewables, vector=g.vector,
            cost_overrides=bundle.cost_overrides,
        )
        if bundle.env_overrides:
            _deep_merge(env_overrides, bundle.env_overrides)

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
        infeasible = ~np.isfinite(arr) | (arr <= c.infeas_threshold)
        arr = np.where(infeasible, np.nan, arr)

        n_total = int(arr.size)
        k_feasible = int(n_total - int(infeasible.sum()))
        mean_score = float(np.nanmean(arr)) if k_feasible >= 1 else float("nan")
        var_score = float(np.nanvar(arr, ddof=1)) if k_feasible >= 2 else float("nan")
        sd_score = float(np.sqrt(var_score)) if np.isfinite(var_score) else 0.0
        objective_value = (
            mean_score - g.stdev_penalty * sd_score
            if np.isfinite(mean_score)
            else float("nan")
        )

        entry = {
            "eval_id": _eval_counter,
            "timestamp": datetime.now().isoformat(),
            "objective": objective_value,
            "mean_score": mean_score,
            "var_score": var_score,
            "num_workers": len(raw_scores),
            "k_feasible": k_feasible,
            "n_total": n_total,
            "feasibility_rate": k_feasible / max(n_total, 1),
            "worker_scores": list(raw_scores),
            "x": [float(v) for v in x_joint],
        }
        flat_dims = flatten_dims(planning_params, env_overrides)
        for key in PARAM_KEYS:
            entry[key] = flat_dims[key]
        entry["capex"] = planning_params["capex"]
        entry["opex"] = planning_params["opex"]
        for name, val in zip(registry.names, theta):
            entry[f"theta.{name}"] = float(val)
        _results_log.append(entry)

        logger.info(
            "kg.evaluation",
            eval_id=_eval_counter,
            objective=objective_value,
            k_feasible=k_feasible,
            n_total=n_total,
            theta=dict(zip(registry.names, (float(v) for v in theta))),
        )
        if _bayesopt_dir is not None:
            save_results(_results_log, _bayesopt_dir, _run_timestamp, registry)
        return arr

    return objective


def load_sobol_cache(sobol_dir: Path, cfg: Config, registry) -> list:
    """Replay `sobol_parametric.py` rows as initial BO observations.

    Rows are matched on vector / bounds_expansion / dynamic_price *and*
    on the theta parameter list — a cache built for a different theta set
    describes a different joint space and must not be mixed in. `x` is
    the joint [x_design | theta] vector. Non-finite worker scores become
    NaN and rows are NaN-padded to `num_instances` so N is consistent.
    """
    g, c = cfg.general, cfg.constrained_bo
    cap = c.n_sobol_cache
    observations, n_missing, n_mismatch, n_theta_mismatch = [], 0, 0, 0

    row_dirs = sorted(sobol_dir.glob("row_*"))
    for rd in row_dirs:
        if cap is not None and len(observations) >= cap:
            break
        files = sorted(rd.glob("result_*.json"))
        if not files:
            n_missing += 1
            continue
        data = json.loads(files[-1].read_text())

        if (data.get("vector") != g.vector
                or float(data.get("bounds_expansion", -1)) != float(g.bounds_expansion)
                or bool(data.get("dynamic_price", False)) != bool(g.dynamic_price)):
            n_mismatch += 1
            continue
        if list(data.get("theta_params") or []) != list(registry.names):
            n_theta_mismatch += 1
            continue

        scores = data.get("worker_scores") or []
        arr = np.asarray([np.nan if v is None else v for v in scores], dtype=float).ravel()
        if arr.size < g.num_instances:
            arr = np.concatenate([arr, np.full(g.num_instances - arr.size, np.nan)])
        arr = np.where(np.isfinite(arr), arr, np.nan)

        x = np.asarray(data["x"], dtype=float)
        if x.size != len(PARAM_KEYS) + registry.dim:
            n_mismatch += 1
            continue
        observations.append((x, arr))

    logger.info(
        "kg.sobol_cache_loaded",
        dir=str(sobol_dir), n_rows_found=len(row_dirs), n_loaded=len(observations),
        n_missing_result=n_missing, n_mismatched=n_mismatch,
        n_theta_mismatched=n_theta_mismatch, cap=cap,
    )
    return observations


def save_results(results_log: list, bayesopt_dir: Path, run_timestamp: str, registry):
    if not results_log:
        return
    fieldnames = [
        "eval_id", "timestamp", "objective", "mean_score", "var_score",
        "num_workers", "k_feasible", "n_total", "feasibility_rate",
    ] + PARAM_KEYS + ["capex", "opex"] + [f"theta.{n}" for n in registry.names]
    csv_path = bayesopt_dir / f"kg_results_{run_timestamp}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for entry in results_log:
            writer.writerow({k: v for k, v in entry.items() if k != "worker_scores"})
    with open(bayesopt_dir / f"kg_results_detailed_{run_timestamp}.json", "w") as f:
        json.dump(results_log, f, indent=2)


def _parse_cli_overrides() -> argparse.Namespace:
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
    g, c = cfg.general, cfg.constrained_bo
    if cfg.theta is None:
        raise SystemExit(
            "config.yml has no `theta:` block — this is the parametric runner. "
            "Use scripts/idc/constrained_bo.py for a design-only run."
        )
    if cfg.kg is None:
        raise SystemExit(
            "config.yml has no `kg:` block — required by the KG runner. "
            "Use scripts/parametric/parametric_bo.py for the EI-based run."
        )

    registry = registry_from_names(cfg.theta.params, vector=g.vector)

    configure_jax_threads(cfg.nlp.n_devices, cfg.nlp.blas_threads)
    from hydro_bo.mpc.dispatcher import ensure_ray  # noqa: E402
    ensure_ray(num_cpus=cfg.general.num_devices)
    from hydro_bo.opt.parametric import KnowledgeGradientBayesopt  # noqa: E402

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
    logger.info("kg.master_seed", master_seed=_master_seed)

    ref_path = planning_model_path(SCRIPTS_DIR, g.vector, g.planning_dir)
    if not ref_path.exists():
        logger.error(
            "kg.missing_planning_model",
            path=str(ref_path),
            message=f"Run: python scripts/idc/planning_model.py {g.vector}",
        )
        return None, None
    with open(ref_path) as f:
        ref = yaml.safe_load(f)

    bounds_x = build_bounds(ref, g.bounds_expansion)
    cat_vars = build_cat_vars(bounds_x)          # design integer dims only
    bounds = np.vstack([bounds_x, registry.bounds()])

    # Fail loudly now if a theta sink no longer resolves, rather than
    # 15 minutes into the first MPC evaluation.
    from h2_plan.data import DefaultParams
    from hydro_bo.envs.shipping.utils import import_mpc_data
    registry.validate_runtime(
        parameter_tree=DefaultParams("default").formulation_parameters,
        mpc_param_names=set(import_mpc_data(dict(ref), g.vector)["params"]),
    )

    logger.info(
        "kg.search_space",
        design_dims=len(PARAM_KEYS), theta_dims=registry.dim, joint_dims=bounds.shape[0],
    )
    for key, (lo, hi) in zip(PARAM_KEYS + registry.names, bounds):
        logger.info("kg.parameter_bounds", parameter=key, lower=lo, upper=hi)

    with open(_bayesopt_dir / "args.json", "w") as f:
        json.dump(
            {
                "general": g.__dict__,
                "constrained_bo": c.__dict__,
                "nlp": cfg.nlp.__dict__,
                "theta": {"params": registry.names, "seed": cfg.theta.seed,
                          "bounds": registry.bounds().tolist()},
                "kg": cfg.kg.__dict__,
                "master_seed_resolved": _master_seed,
            },
            f, indent=2, default=str,
        )

    if cfg.nlp.sqp_use_exact_hessian:
        logger.warning(
            "kg.outer_exact_hessian",
            message=("nlp.sqp_use_exact_hessian is true; the outer solve would "
                     "form Hessians through the nested inner solves. Set it "
                     "false for a limited-memory update."),
        )

    bo = KnowledgeGradientBayesopt(
        f=_objective_factory(cfg, ref, registry),
        bounds=bounds,
        d_theta=registry.dim,
        kg_args=cfg.kg.to_kg_args(),
        n_initial_points=c.n_initial_points,
        iter_limit=c.iter_budget,
        lam=g.stdev_penalty,
        n_restarts=cfg.nlp.acq_n_restarts,
        pow_sobol=cfg.nlp.acq_pow_sobol,
        seed=_master_seed % (2**31),
        cat_vars=cat_vars,
        sqp_config=cfg.nlp.to_sqp_config(),
        pad_initial=cfg.nlp.pad_initial,
        gp_lbfgs_max_iter=cfg.nlp.gp_lbfgs_max_iter,
        gp_mu_kernel=cfg.nlp.gp_mu_kernel,
        gp_log_var_kernel=cfg.nlp.gp_log_var_kernel,
    )

    sobol_dir = resolve_sobol_dir(c.sobol_dir, SCRIPTS_DIR, g.vector)
    if sobol_dir is not None:
        if not sobol_dir.exists():
            logger.warning("kg.sobol_dir_missing", path=str(sobol_dir))
        else:
            preloaded = load_sobol_cache(sobol_dir, cfg, registry)
            for x, samples in preloaded:
                bo.observe(x, samples)
            if preloaded:
                bo.n_initial_points = len(preloaded)
                logger.info("kg.sobol_phase_skipped_via_cache",
                            n_preloaded=len(preloaded))

    best_x, best_score = bo.run()
    best_design, best_theta = best_x[:len(PARAM_KEYS)], best_x[len(PARAM_KEYS):]
    best_planning_params, best_env_overrides = params_from_x(
        best_design, ref, renewables=g.renewables, vector=g.vector,
        cost_overrides=registry.apply(best_theta).cost_overrides,
    )
    best_flat = flatten_dims(best_planning_params, best_env_overrides)

    logger.info("kg.complete")
    logger.info("kg.best_score", score=best_score)
    for k, v in best_flat.items():
        logger.info("parametric.parameter", name=k, value=v)
    for name, val in zip(registry.names, best_theta):
        logger.info("kg.theta", name=name, value=float(val))

    snapshot = dict(best_planning_params)
    snapshot["wind_forecast_mean"] = best_flat["wind_forecast_mean"]
    snapshot["theta"] = dict(zip(registry.names, (float(v) for v in best_theta)))
    out_path = _bayesopt_dir / f"{g.vector}-Chile-kg.yml"
    with open(out_path, "w") as f:
        yaml.dump(snapshot, f, default_flow_style=False)
    logger.info("kg.saved", path=str(out_path))

    save_results(_results_log, _bayesopt_dir, _run_timestamp, registry)
    return snapshot, best_score


if __name__ == "__main__":
    main()
