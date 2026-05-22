"""Ablation study over the BO constraint level × initial-design density.

Each PBS array index maps to one BO run via:

    cells_per_mode  = len(DENSITIES) * N_BATCHES         # 4 * 25 = 100
    constraint_idx  = array_index // cells_per_mode      # 0..2
    rem             = array_index %  cells_per_mode
    density_idx     = rem // N_BATCHES                   # 0..3
    batch_id        = rem %  N_BATCHES                   # 0..24

    CONSTRAINT_MODES = ("none", "ci50", "ci100")
    DENSITIES        = (8, 16, 32, 64)

Total: 3 * 4 * 25 = 300 cells.

Every cell uses an initial design pulled from a shared manifest JSON:
the manifest fixes 25 random batches of 64 row indices into the
configured sobol_dir, and density-D = the first D rows of a batch
(nested). Each run is BO with the same `iter_budget` (default 25)
appended to that initial design.

The three constraint modes all run through `ConstrainedBayesopt`. The
"none" arm uses a thin subclass that disables the binomial GP fit and
the feasibility filter (`_UnconstrainedAblationBO`), so the only
difference between arms is the chance constraint itself.

Output directory:
    scripts/tmp/ablation/<run-id>/<VECTOR>/sobol_NN_CI_XXX_comb_BB/

Two CLI sub-commands:

    python ablation_bo.py generate-manifest --vector NH3 --out PATH
        Pre-computes the manifest. Called once by the shell submitter
        before qsub'ing the array.

    python ablation_bo.py run --array-index I --array-size 300 \
        --run-id RID --manifest-path PATH [--vector V] [--ncpus N] \
        [--iter-budget 25]
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
from scipy.special import ndtri

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

# Sweep axes. Order matters — the decode_index function below assumes
# this exact layout, and so does the folder-naming.
CONSTRAINT_MODES = ("none", "ci50", "ci100")
DENSITIES = (8, 16, 32, 64)
N_BATCHES = 25

# z_sc for each constrained mode. "none" doesn't use this (constraint
# is disabled at the BO-class level); we still pass a finite placeholder
# so the BO constructor doesn't choke on NaN.
_CI_PCT = {"ci50": 50.0, "ci100": 99.5}

# Per-run mutable state. Module-level so the objective closure can write
# to the results log without threading another arg through.
_eval_counter: int = 0
_results_log: list = []
_bayesopt_dir: Optional[Path] = None
_run_timestamp: str = ""
_master_seed: int = 0


# ---------------------------------------------------------------------------
# Index decoding & label helpers
# ---------------------------------------------------------------------------

def decode_index(array_index: int) -> tuple[str, int, int]:
    """Map array_index in [0, 300) -> (constraint_mode, density, batch_id)."""
    cells_per_mode = len(DENSITIES) * N_BATCHES
    if array_index < 0 or array_index >= N_BATCHES * len(DENSITIES) * len(CONSTRAINT_MODES):
        raise ValueError(
            f"array_index {array_index} out of range "
            f"[0, {N_BATCHES * len(DENSITIES) * len(CONSTRAINT_MODES)})"
        )
    constraint_idx, rem = divmod(array_index, cells_per_mode)
    density_idx, batch_id = divmod(rem, N_BATCHES)
    return CONSTRAINT_MODES[constraint_idx], DENSITIES[density_idx], batch_id


def z_sc_for_mode(mode: str) -> float:
    """z-score derived from CI for the constrained modes. 'none' returns 0.0
    (placeholder — the constraint is disabled at the BO-class level)."""
    if mode == "none":
        return 0.0
    return float(ndtri(_CI_PCT[mode] / 100.0))


def task_label(mode: str, density: int, batch_id: int) -> str:
    """sobol_NN_CI_XXX_comb_BB style label. CI=NONE for the unconstrained arm."""
    if mode == "none":
        ci_label = "NONE"
    else:
        ci_label = f"{int(round(_CI_PCT[mode])):03d}"
    return f"sobol_{density:02d}_CI_{ci_label}_comb_{batch_id:02d}"


# ---------------------------------------------------------------------------
# Manifest generation & loading
# ---------------------------------------------------------------------------

def _scan_sobol_dir_for_eligibility(
    sobol_dir: Path,
    cfg: Config,
    infeas_threshold: float,
) -> list[tuple[int, int]]:
    """Walk every `row_*` under `sobol_dir` and return [(row_idx, n_feasible)].

    Rows that don't match the active (vector, bounds_expansion,
    dynamic_price) triple, or that have no `result_*.json` written yet,
    are dropped. `n_feasible` is the count of worker scores that are
    finite AND > `infeas_threshold` — same definition the constrained BO
    uses to decide what counts as a usable initial-design observation."""
    g = cfg.general
    eligible: list[tuple[int, int]] = []
    n_missing = n_mismatched = 0
    for rd in sorted(sobol_dir.glob("row_*")):
        try:
            row_idx = int(rd.name.split("_")[1])
        except (IndexError, ValueError):
            continue
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
        n_feas = int(np.sum(np.isfinite(arr) & (arr > infeas_threshold)))
        eligible.append((row_idx, n_feas))
    logger.info(
        "ablation.eligibility_scan",
        sobol_dir=str(sobol_dir),
        n_eligible=len(eligible),
        n_missing_result=n_missing,
        n_mismatched=n_mismatched,
    )
    return eligible


def generate_manifest(
    sobol_dir: Path,
    cfg: Config,
    infeas_threshold: float,
    out_path: Path,
    n_batches: int = N_BATCHES,
    max_density: int = max(DENSITIES),
    min_density: int = min(DENSITIES),
    min_feasible: int = 2,
    seed: int = 0,
) -> dict:
    """Sample `n_batches` batches of `max_density` row indices.

    Rejection rule: the first `min_density` rows of each batch must
    contain at least `min_feasible` feasible observations. With
    `min_density=8` this also implies every nested slice (16, 32, 64) is
    at least as feasible, so we only need to check the smallest.

    Deterministic in `seed` modulo Sobol-pool ordering on disk.
    Writes JSON to `out_path`. Raises if the eligible pool is too small
    or if `n_batches` valid batches can't be sampled in a reasonable
    number of attempts (suggesting the pool is too infeasible-heavy)."""
    g = cfg.general
    eligible = _scan_sobol_dir_for_eligibility(sobol_dir, cfg, infeas_threshold)
    if len(eligible) < max_density:
        raise RuntimeError(
            f"Not enough eligible sobol rows under {sobol_dir}: "
            f"have {len(eligible)}, need at least {max_density}. "
            f"Generate more sobol points first with sobol_mpc.py."
        )

    rng = np.random.default_rng(seed)
    batches: list[list[int]] = []
    n_rejected = 0
    max_attempts = n_batches * 200  # plenty of headroom

    for _ in range(max_attempts):
        if len(batches) >= n_batches:
            break
        # Sample max_density distinct indices into `eligible`.
        chosen = rng.choice(len(eligible), size=max_density, replace=False)
        # Count feasibles in the smallest density slice; nested batches
        # mean larger slices automatically pass.
        n_feas_in_slice = sum(int(eligible[i][1] > 0) for i in chosen[:min_density])
        if n_feas_in_slice >= min_feasible:
            batches.append([int(eligible[i][0]) for i in chosen])
        else:
            n_rejected += 1

    if len(batches) < n_batches:
        raise RuntimeError(
            f"Could not generate {n_batches} valid batches after "
            f"{max_attempts} attempts (rejected={n_rejected}, "
            f"min_density={min_density}, min_feasible={min_feasible}). "
            f"The sobol pool may be too infeasible-heavy — generate more "
            f"sobol points or loosen the rejection rule."
        )

    manifest = {
        "seed": int(seed),
        "n_batches": int(n_batches),
        "max_density": int(max_density),
        "min_density": int(min_density),
        "min_feasible": int(min_feasible),
        "infeas_threshold": float(infeas_threshold),
        "sobol_dir": str(sobol_dir),
        "vector": g.vector,
        "bounds_expansion": float(g.bounds_expansion),
        "dynamic_price": bool(g.dynamic_price),
        "n_rejected": int(n_rejected),
        "n_eligible_rows": int(len(eligible)),
        # list of len n_batches, each a list[int] of length max_density.
        "batches": batches,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(
        "ablation.manifest_saved",
        path=str(out_path),
        n_batches=len(batches),
        n_rejected=n_rejected,
        n_eligible=len(eligible),
    )
    return manifest


def load_manifest_rows(
    sobol_dir: Path,
    cfg: Config,
    infeas_threshold: float,
    row_indices: list[int],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load (x, samples) tuples for the listed row indices in order.

    `samples` is the per-worker score array with non-finite /
    catastrophic entries set to NaN — same shape the constrained BO's
    `_objective_factory` produces on a live evaluation, so the BO sees
    initial-design rows and online observations as one continuous
    stream. Rows that fail to load are logged and skipped — the BO may
    end up with fewer than `len(row_indices)` initial observations."""
    g = cfg.general
    observations: list[tuple[np.ndarray, np.ndarray]] = []
    n_missing = n_mismatched = 0
    for idx in row_indices:
        rd = sobol_dir / f"row_{idx:05d}"
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
        "ablation.manifest_rows_loaded",
        n_requested=len(row_indices),
        n_loaded=len(observations),
        n_missing_result=n_missing,
        n_mismatched=n_mismatched,
    )
    return observations


# ---------------------------------------------------------------------------
# Objective closure & results IO
# ---------------------------------------------------------------------------

def _objective_factory(cfg: Config, ref: dict, infeas_threshold: float):
    """Identical shape to array_constrained_bo._objective_factory:
    runs N_INSTANCES MPC simulations and returns the per-worker sample
    array with non-finite / catastrophic entries marked NaN. The
    ConstrainedBayesopt Dataset handles NaN samples natively for all
    three arms."""
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
            "ablation.evaluation",
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


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------

def _cmd_generate_manifest(cli) -> None:
    """Pre-compute the shared sobol-batch manifest. Invoked once by the
    submitter shell before qsub'ing the array; the resulting JSON path
    is then threaded through to every PBS task via `--manifest-path`."""
    cfg = load_config(
        SCRIPTS_DIR / "config.yml",
        vector_override=cli.vector,
        num_devices_override=None,
    )
    g, c = cfg.general, cfg.constrained_bo

    sobol_dir = resolve_sobol_dir(c.sobol_dir, SCRIPTS_DIR, g.vector)
    if sobol_dir is None or not sobol_dir.exists():
        raise RuntimeError(
            f"sobol_dir does not exist: {sobol_dir}. "
            f"Run sobol_mpc.py to populate it before generating a manifest."
        )

    out_path = Path(cli.out)
    if not out_path.is_absolute():
        out_path = SCRIPTS_DIR / out_path

    configure_logging(
        log_file=out_path.parent / "manifest_generation.log",
        package_level=g.log_level,
    )
    logger.info(
        "ablation.manifest_args",
        vector=g.vector,
        sobol_dir=str(sobol_dir),
        out=str(out_path),
        n_batches=cli.n_batches,
        max_density=cli.max_density,
        min_density=cli.min_density,
        min_feasible=cli.min_feasible,
        seed=cli.seed,
        infeas_threshold=c.infeas_threshold,
    )

    generate_manifest(
        sobol_dir=sobol_dir,
        cfg=cfg,
        infeas_threshold=c.infeas_threshold,
        out_path=out_path,
        n_batches=cli.n_batches,
        max_density=cli.max_density,
        min_density=cli.min_density,
        min_feasible=cli.min_feasible,
        seed=cli.seed,
    )


def _cmd_run(cli) -> None:
    """Run a single ablation cell. PBS array entry point."""
    cfg = load_config(
        SCRIPTS_DIR / "config.yml",
        vector_override=cli.vector,
        num_devices_override=cli.ncpus,
    )
    g, c = cfg.general, cfg.constrained_bo

    expected_size = len(CONSTRAINT_MODES) * len(DENSITIES) * N_BATCHES
    if cli.array_size != expected_size:
        raise ValueError(
            f"--array-size {cli.array_size} != expected {expected_size} "
            f"({len(CONSTRAINT_MODES)} modes × {len(DENSITIES)} densities × "
            f"{N_BATCHES} batches)"
        )

    mode, density, batch_id = decode_index(cli.array_index)
    label = task_label(mode, density, batch_id)

    # Start Ray BEFORE JAX threads spawn. JAX/XLA spawn worker threads
    # at first use, and Ray's raylet/GCS startup forks subprocesses —
    # forking a multi-threaded parent deadlocks the children. Once Ray
    # is up its raylet/GCS are independent processes that don't care
    # about our thread state.
    ensure_ray(num_cpus=g.num_devices)
    configure_jax_threads(cfg.nlp.n_devices, cfg.nlp.blas_threads)
    # BO classes pull jax at import — defer until after configure_jax_threads.
    from hydro_bo.opt import ConstrainedBayesopt, MeanVarBayesopt  # noqa: E402

    class _UnconstrainedAblationBO(ConstrainedBayesopt):
        """ConstrainedBayesopt with the chance constraint switched off.

        Only the constraint differs from the constrained arms — mean /
        log-var GP fitting, acquisition, and the SQP solver are the same
        machinery. Two overrides:

          * `_fit_surrogates` skips the binomial GP fit, so the
            constraint surface is never built.
          * `_check_stuck_risk` always sets `_stuck_skip = True`, which
            steers `_build_acquisition` / `_build_solver` onto the
            unconstrained-EI / MixedIntNLP branches inside the parent.
          * `_best_observed` falls back to MeanVar's incumbent rule
            (best mu - λ·sd), without the `k/N >= p_targ` filter.
        """

        def _fit_surrogates(self) -> None:
            MeanVarBayesopt._fit_surrogates(self)
            self._stuck_skip = True

        def _check_stuck_risk(self) -> None:
            self._stuck_skip = True

        def _best_observed(self):
            return MeanVarBayesopt._best_observed(self)

    global _eval_counter, _results_log, _bayesopt_dir, _run_timestamp, _master_seed
    _eval_counter = 0
    _results_log = []

    now = datetime.now()
    _run_timestamp = now.strftime("%Y%m%d_%H%M%S")
    _bayesopt_dir = (
        SCRIPTS_DIR / "tmp" / "ablation" / cli.run_id / g.vector / label
    )
    _bayesopt_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(log_file=_bayesopt_dir / "run.log", package_level=g.log_level)

    _master_seed = resolve_master_seed(g.master_seed)
    logger.info(
        "ablation.cell",
        run_id=cli.run_id,
        array_index=cli.array_index,
        array_size=cli.array_size,
        mode=mode,
        density=density,
        batch_id=batch_id,
        label=label,
        master_seed=_master_seed,
    )

    # Load the manifest and pull this batch's first `density` row indices.
    manifest_path = Path(cli.manifest_path)
    with open(manifest_path) as f:
        manifest = json.load(f)
    if manifest.get("vector") != g.vector:
        raise RuntimeError(
            f"Manifest vector {manifest.get('vector')!r} != cfg vector {g.vector!r}"
        )
    if batch_id >= len(manifest["batches"]):
        raise RuntimeError(
            f"batch_id {batch_id} out of manifest range {len(manifest['batches'])}"
        )
    if density > int(manifest["max_density"]):
        raise RuntimeError(
            f"density {density} > manifest max_density {manifest['max_density']}"
        )
    row_indices = list(manifest["batches"][batch_id][:density])

    # Snapshot resolved config + decoded ablation knobs for reproducibility.
    iter_budget = int(cli.iter_budget)
    z_sc = z_sc_for_mode(mode)
    with open(_bayesopt_dir / "args.json", "w") as f:
        json.dump(
            {
                "general": g.__dict__,
                "constrained_bo": c.__dict__,
                "nlp": cfg.nlp.__dict__,
                "ablation": {
                    "run_id": cli.run_id,
                    "array_index": cli.array_index,
                    "array_size": cli.array_size,
                    "mode": mode,
                    "density": density,
                    "batch_id": batch_id,
                    "label": label,
                    "iter_budget": iter_budget,
                    "z_sc": z_sc,
                    "p_targ": c.p_targ,
                    "manifest_path": str(manifest_path),
                    "row_indices": row_indices,
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
            "ablation.missing_planning_model",
            path=str(ref_path),
            message=f"Run: python scripts/planning.py {g.vector}",
        )
        return
    with open(ref_path) as f:
        ref = yaml.safe_load(f)

    bounds = build_bounds(ref, g.bounds_expansion)
    cat_vars = build_cat_vars(bounds)
    logger.info("ablation.search_space", dims=len(PARAM_KEYS))

    bo_cls = _UnconstrainedAblationBO if mode == "none" else ConstrainedBayesopt
    bo = bo_cls(
        f=_objective_factory(cfg, ref, c.infeas_threshold),
        bounds=bounds,
        n_initial_points=density,
        iter_limit=iter_budget,
        lam=g.stdev_penalty,
        n_restarts=cfg.nlp.acq_n_restarts,
        pow_sobol=cfg.nlp.acq_pow_sobol,
        seed=_master_seed % (2**31),
        cat_vars=cat_vars,
        p_targ=c.p_targ,
        z_sc=z_sc,
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
    preloaded = load_manifest_rows(sobol_dir, cfg, c.infeas_threshold, row_indices)
    for x, samples in preloaded:
        bo.observe(x, samples)
    if preloaded:
        # Identical to the array_constrained_bo cache-skip pattern: when
        # the initial design is fully preloaded, the BO's internal Sobol
        # phase is collapsed and it moves straight to the acquisition loop.
        bo.n_initial_points = len(preloaded)
        logger.info(
            "ablation.sobol_phase_skipped_via_cache", n_preloaded=len(preloaded)
        )

    best_x, best_score = bo.run()
    best_planning_params, best_env_overrides = params_from_x(
        best_x, ref, renewables=g.renewables, vector=g.vector
    )
    best_flat = flatten_dims(best_planning_params, best_env_overrides)

    logger.info("ablation.complete", mode=mode, density=density, batch_id=batch_id)
    logger.info("ablation.best_score", score=best_score)
    for k, v in best_flat.items():
        logger.info("ablation.parameter", name=k, value=v)

    out_path = _bayesopt_dir / f"{g.vector}-{label}-best.yml"
    snapshot = dict(best_planning_params)
    snapshot["wind_forecast_mean"] = best_flat["wind_forecast_mean"]
    with open(out_path, "w") as f:
        yaml.dump(snapshot, f, default_flow_style=False)
    logger.info("ablation.saved", path=str(out_path))

    save_results(_results_log, _bayesopt_dir, _run_timestamp)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=True)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_gen = sub.add_parser(
        "generate-manifest",
        help="Pre-compute the shared sobol-batch manifest. Called once by "
             "the shell submitter before qsub'ing the array.",
    )
    p_gen.add_argument("--vector", type=str, default=None,
                       help="Hydrogen vector; overrides general.vector.")
    p_gen.add_argument("--out", type=str, required=True,
                       help="Output manifest JSON path. Relative paths "
                            "are interpreted relative to scripts/.")
    p_gen.add_argument("--n-batches", type=int, default=N_BATCHES)
    p_gen.add_argument("--max-density", type=int, default=max(DENSITIES))
    p_gen.add_argument("--min-density", type=int, default=min(DENSITIES))
    p_gen.add_argument("--min-feasible", type=int, default=2,
                       help="Minimum feasibles in the smallest-density slice "
                            "for a batch to be accepted.")
    p_gen.add_argument("--seed", type=int, default=0,
                       help="Sampler seed; deterministic given the on-disk "
                            "sobol_dir ordering.")

    p_run = sub.add_parser(
        "run", help="Single ablation cell. PBS array entry point.",
    )
    p_run.add_argument("--array-index", type=int, required=True,
                       help="0-based PBS_ARRAY_INDEX.")
    p_run.add_argument("--array-size", type=int, required=True,
                       help="Total cells in the array; must equal "
                            f"{len(CONSTRAINT_MODES) * len(DENSITIES) * N_BATCHES}.")
    p_run.add_argument("--run-id", type=str, required=True,
                       help="Shared parent dir id (submission timestamp).")
    p_run.add_argument("--manifest-path", type=str, required=True,
                       help="Path to the manifest JSON produced by "
                            "generate-manifest.")
    p_run.add_argument("--iter-budget", type=int, default=25,
                       help="BO iterations *after* the cached sobol design "
                            "(default 25).")
    p_run.add_argument("--vector", type=str, default=None,
                       help="Hydrogen vector; overrides general.vector.")
    p_run.add_argument("--ncpus", type=int, default=None,
                       help="Override general.num_devices.")
    return parser


def main():
    parser = _build_parser()
    cli = parser.parse_args()
    if cli.cmd == "generate-manifest":
        _cmd_generate_manifest(cli)
    elif cli.cmd == "run":
        _cmd_run(cli)
    else:
        parser.error(f"unknown cmd {cli.cmd!r}")


if __name__ == "__main__":
    main()
