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


@dataclass(frozen=True)
class SobolCfg:
    pow_n: int
    seed: int
    walltime_seconds: Optional[int]
    buffer_seconds: int


@dataclass(frozen=True)
class UnconstrainedCfg:
    iter_budget: int
    n_initial_points: int
    failure_penalty: float
    min_valid_samples: int
    sobol_dir: Optional[str]


@dataclass(frozen=True)
class ConstrainedCfg:
    iter_budget: int
    n_initial_points: int
    infeas_threshold: float
    p_targ: float
    z_sc: float
    l1_penalty: float
    sobol_dir: Optional[str]


@dataclass(frozen=True)
class Config:
    general: GeneralCfg
    sobol: SobolCfg
    unconstrained_bo: UnconstrainedCfg
    constrained_bo: ConstrainedCfg


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
        ),
        unconstrained_bo=UnconstrainedCfg(
            iter_budget=int(raw["unconstrained_bo"]["iter_budget"]),
            n_initial_points=int(raw["unconstrained_bo"]["n_initial_points"]),
            failure_penalty=float(raw["unconstrained_bo"]["failure_penalty"]),
            min_valid_samples=int(raw["unconstrained_bo"]["min_valid_samples"]),
            sobol_dir=raw["unconstrained_bo"].get("sobol_dir"),
        ),
        constrained_bo=ConstrainedCfg(
            iter_budget=int(raw["constrained_bo"]["iter_budget"]),
            n_initial_points=int(raw["constrained_bo"]["n_initial_points"]),
            infeas_threshold=float(raw["constrained_bo"]["infeas_threshold"]),
            p_targ=float(raw["constrained_bo"]["p_targ"]),
            z_sc=float(raw["constrained_bo"]["z_sc"]),
            l1_penalty=float(raw["constrained_bo"]["l1_penalty"]),
            sobol_dir=raw["constrained_bo"].get("sobol_dir"),
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
