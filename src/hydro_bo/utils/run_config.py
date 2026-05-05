"""Loader for the shared run-script config.yml.

Single source of truth for the `sobol_mpc`, `unconstrained_bo`, and
`constrained_bo` scripts: each one calls `load_config(path)` at the top
of `main()` and reads everything else from the returned `Config`
dataclass — no argparse, no CAPS-block scattered across the script.

The YAML lives next to the scripts that use it (typically
`scripts/config.yml`); this module is in `hydro_bo.utils` so other
non-script consumers can also load and inspect it.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass(frozen=True)
class GeneralCfg:
    vector: str
    renewables: str
    weather_file: List[str]
    bounds_expansion: float
    num_devices: int
    num_instances: int
    timeout: int
    forecast_horizon: int
    master_seed: Optional[int]
    dynamic_price: bool
    stdev_penalty: float
    log_level: int  # stdlib logging level for the hydro_bo namespace.


@dataclass(frozen=True)
class SobolCfg:
    pow_n: int
    seed: int
    walltime_seconds: Optional[int]
    buffer_seconds: int
    sobol_dir: Optional[str]  # output directory for sobol_mpc; supports `{vector}`.


@dataclass(frozen=True)
class NlpCfg:
    """Solver settings: septal SQP for acquisition optimisation, jaxopt
    L-BFGS for GP fits.

    `n_max` is the padded shape used for GP observations — set generously
    above the expected total count (n_initial + iter_budget +
    n_sobol_cache). All GP code traces once at this shape, so the JIT
    cache persists across BO iterations.
    """
    acq_pow_sobol: int
    acq_n_restarts: int
    pad_initial: int
    gp_lbfgs_max_iter: int
    n_devices: int
    blas_threads: int
    sqp_max_iter: int
    sqp_tol_stationarity: float
    sqp_tol_feasibility: float
    sqp_use_exact_hessian: bool

    def to_sqp_config(self):
        """Build a `septal.jax.sqp.SQPConfig` from these settings."""
        from septal.jax.sqp import SQPConfig
        return SQPConfig(
            max_iter=self.sqp_max_iter,
            use_exact_hessian=self.sqp_use_exact_hessian,
            tol_stationarity=self.sqp_tol_stationarity,
            tol_feasibility=self.sqp_tol_feasibility,
        )


@dataclass(frozen=True)
class UnconstrainedCfg:
    iter_budget: int
    n_initial_points: int
    failure_penalty: float
    min_valid_samples: int
    sobol_dir: Optional[str]
    n_sobol_cache: Optional[int]  # cap on cached Sobol rows loaded (None = all).


@dataclass(frozen=True)
class ConstrainedCfg:
    iter_budget: int
    n_initial_points: int
    infeas_threshold: float
    p_targ: float
    z_sc: float
    l1_penalty: float
    sobol_dir: Optional[str]
    n_sobol_cache: Optional[int]  # cap on cached Sobol rows loaded (None = all).


@dataclass(frozen=True)
class Config:
    general: GeneralCfg
    sobol: SobolCfg
    nlp: NlpCfg
    unconstrained_bo: UnconstrainedCfg
    constrained_bo: ConstrainedCfg


def _optional_int(raw) -> Optional[int]:
    if raw is None:
        return None
    return int(raw)


def _resolve_n_devices(raw) -> int:
    """null → os.cpu_count(); otherwise the supplied int."""
    if raw is None:
        return max(1, os.cpu_count() or 1)
    return int(raw)


def _resolve_log_level(raw) -> int:
    if raw is None:
        return logging.INFO
    if isinstance(raw, int):
        return raw
    name = str(raw).strip().upper()
    level = logging.getLevelName(name)
    if not isinstance(level, int):
        raise ValueError(
            f"Unknown log_level {raw!r}; expected DEBUG, INFO, WARNING, ERROR, CRITICAL."
        )
    return level


def _resolve_general(g: dict) -> GeneralCfg:
    nd = g.get("num_devices")
    if nd is None:
        nd = max(1, (os.cpu_count() or 1) - 1)
    ni = g.get("num_instances")
    if ni is None:
        ni = nd
    return GeneralCfg(
        vector=str(g["vector"]),
        renewables=str(g["renewables"]),
        weather_file=list(g["weather_file"]),
        bounds_expansion=float(g["bounds_expansion"]),
        num_devices=int(nd),
        num_instances=int(ni),
        timeout=int(g["timeout"]),
        forecast_horizon=int(g["forecast_horizon"]),
        master_seed=g.get("master_seed"),
        dynamic_price=bool(g.get("dynamic_price", False)),
        stdev_penalty=float(g["stdev_penalty"]),
        log_level=_resolve_log_level(g.get("log_level")),
    )


def load_config(path: str | Path, *, vector_override: Optional[str] = None) -> Config:
    """Load the YAML at `path` and return a typed `Config`. Caller is
    responsible for the path (typically `scripts/config.yml`).

    `vector_override`, if given, replaces `general.vector` before the
    rest of the config is built — used by the run scripts to expose a
    single `--vector` CLI flag while keeping every other knob in YAML.
    """
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if vector_override is not None:
        raw["general"]["vector"] = str(vector_override)
    return Config(
        general=_resolve_general(raw["general"]),
        sobol=SobolCfg(
            pow_n=int(raw["sobol"]["pow_n"]),
            seed=int(raw["sobol"]["seed"]),
            walltime_seconds=raw["sobol"].get("walltime_seconds"),
            buffer_seconds=int(raw["sobol"].get("buffer_seconds", 300)),
            sobol_dir=raw["sobol"].get("sobol_dir"),
        ),
        nlp=NlpCfg(
            acq_pow_sobol=int(raw["nlp"]["acq_pow_sobol"]),
            acq_n_restarts=int(raw["nlp"]["acq_n_restarts"]),
            pad_initial=int(raw["nlp"]["pad_initial"]),
            gp_lbfgs_max_iter=int(raw["nlp"]["gp_lbfgs_max_iter"]),
            n_devices=_resolve_n_devices(raw["nlp"].get("n_devices")),
            blas_threads=int(raw["nlp"].get("blas_threads", 1)),
            sqp_max_iter=int(raw["nlp"]["sqp_max_iter"]),
            sqp_tol_stationarity=float(raw["nlp"]["sqp_tol_stationarity"]),
            sqp_tol_feasibility=float(raw["nlp"]["sqp_tol_feasibility"]),
            sqp_use_exact_hessian=bool(raw["nlp"]["sqp_use_exact_hessian"]),
        ),
        unconstrained_bo=UnconstrainedCfg(
            iter_budget=int(raw["unconstrained_bo"]["iter_budget"]),
            n_initial_points=int(raw["unconstrained_bo"]["n_initial_points"]),
            failure_penalty=float(raw["unconstrained_bo"]["failure_penalty"]),
            min_valid_samples=int(raw["unconstrained_bo"]["min_valid_samples"]),
            sobol_dir=raw["unconstrained_bo"].get("sobol_dir"),
            n_sobol_cache=_optional_int(raw["unconstrained_bo"].get("n_sobol_cache")),
        ),
        constrained_bo=ConstrainedCfg(
            iter_budget=int(raw["constrained_bo"]["iter_budget"]),
            n_initial_points=int(raw["constrained_bo"]["n_initial_points"]),
            infeas_threshold=float(raw["constrained_bo"]["infeas_threshold"]),
            p_targ=float(raw["constrained_bo"]["p_targ"]),
            z_sc=float(raw["constrained_bo"]["z_sc"]),
            l1_penalty=float(raw["constrained_bo"]["l1_penalty"]),
            sobol_dir=raw["constrained_bo"].get("sobol_dir"),
            n_sobol_cache=_optional_int(raw["constrained_bo"].get("n_sobol_cache")),
        ),
    )


def planning_model_path(scripts_dir: Path, vector: str) -> Path:
    """Standard planning-model location, anchored at the supplied
    scripts directory (typically the directory holding config.yml)."""
    return Path(scripts_dir) / "tmp" / "planning" / f"{vector}-Chile.yml"


def env_args_from(cfg: Config) -> dict:
    """The env_args dict consumed by RayMultiMPC, derived from cfg."""
    return {
        "config": {
            "vector": cfg.general.vector,
            "mpc": {"planning_model": f"{cfg.general.vector}-Chile.yml"},
            "weather_data": {"weather_file": cfg.general.weather_file},
            "price_dynamics": {"enabled": cfg.general.dynamic_price},
            "Time": {"forecast_horizon": cfg.general.forecast_horizon},
        },
    }


def merge_env_overrides(cfg: Config, env_overrides: dict) -> dict:
    """Build a fresh env_args from cfg and deep-merge BO-supplied
    `env_overrides` into its `config` block. Caller passes the result
    straight to `RayMultiMPC(env_args=...)` — keeps per-eval overrides
    out of the shared cfg-derived dict."""
    env_args = env_args_from(cfg)

    def _deep_merge(target: dict, src: dict) -> None:
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(target.get(k), dict):
                _deep_merge(target[k], v)
            else:
                target[k] = v

    if env_overrides:
        _deep_merge(env_args["config"], env_overrides)
    return env_args


def resolve_sobol_dir(
    cfg_sobol_dir: Optional[str],
    scripts_dir: Path,
    vector: str,
) -> Optional[Path]:
    """Resolve the optional `sobol_dir` config entry.

    The string may contain a `{vector}` placeholder that is filled in
    with the active vector — useful for shared YAML where you want
    `tmp/sobol/{vector}` rather than per-vector duplicates. Relative
    paths are interpreted relative to the supplied scripts directory.
    """
    if not cfg_sobol_dir:
        return None
    formatted = str(cfg_sobol_dir).format(vector=vector)
    p = Path(formatted)
    if not p.is_absolute():
        p = Path(scripts_dir) / p
    return p
