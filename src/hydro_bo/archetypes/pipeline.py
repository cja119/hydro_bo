"""Shared plumbing for the archetype Sobol array, the per-problem BO and the profiling run: config, Sobol rows, point
evaluation and result files.

Result files (one JSON per evaluated point) carry `x` (original units, SEARCH_KEYS order), `samples_bn` (per-seed NPV
[USD bn], NaN for failed rollouts; this is what the BO observes), the design, per-seed diagnostics and summary
statistics. The BO loads every result file of its problem (Sobol and its own earlier iterations), so a job can be
resubmitted and carries on.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.stats.qmc import Sobol

from hydro_bo.archetypes.rollout import evaluate
from hydro_bo.archetypes.scenarios import SEARCH_KEYS, Problem, design_from_x, load_problems

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "scripts" / "config_archetypes.yml"
DIAGNOSTICS = ("npv", "net", "contract", "sales", "purchases", "linear", "impact", "charter", "port_fees", "stock",
               "landed_kt", "sold_kt", "bought_kt", "produced_kt", "unsold_share", "ship_days_held", "flared_kt",
               "time_limited_days", "wall_s")


def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG) as f:
        return yaml.safe_load(f)


def problems(cfg: dict) -> list[Problem]:
    return load_problems(cfg)


def problem_by_key(cfg: dict, key: str) -> Problem:
    for p in problems(cfg):
        if p.key == key:
            return p
    raise KeyError(f"no problem {key!r}; known: {[p.key for p in problems(cfg)]}")


def sobol_unit(cfg: dict, n: int) -> np.ndarray:
    """First n rows of the scrambled Sobol sequence over the unit cube of SEARCH_KEYS (same for every problem)."""
    m = int(np.ceil(np.log2(max(n, 1))))
    return Sobol(d=len(SEARCH_KEYS), scramble=True, seed=int(cfg["sobol"]["seed"])).random(2**m)[:n]


def task_point(cfg: dict, index: int) -> tuple[Problem, int, np.ndarray, np.ndarray]:
    """Array index -> (problem, Sobol row, unit sample, x in original units). Index = problem_idx * n + row."""
    n = int(cfg["sobol"]["n_per_problem"])
    probs = problems(cfg)
    if not 0 <= index < n * len(probs):
        raise IndexError(f"array index {index} outside [0, {n * len(probs)})")
    problem, row = probs[index // n], index % n
    unit = sobol_unit(cfg, n)[row]
    b = problem.bounds()
    return problem, row, unit, b[:, 0] + unit * (b[:, 1] - b[:, 0])


def seeds(cfg: dict) -> list[int]:
    e = cfg["evaluation"]
    return list(range(int(e["seed_offset"]), int(e["seed_offset"]) + int(e["n_seeds"])))


def evaluate_point(cfg: dict, problem: Problem, x: np.ndarray, ncpus: int, npz_dir: str | Path | None = None) -> dict:
    """Run every seed of one design and summarise. `samples_bn` is the BO observation."""
    e = cfg["evaluation"]
    design = design_from_x(x, problem)
    t0 = time.time()
    runs = evaluate(design, problem, seeds(cfg), ncpus, save_dir=str(npz_dir) if (npz_dir and e.get("save_npz")) else None)
    npv = np.array([r["npv"] for r in runs], dtype=float) / 1000.0
    valid = np.isfinite(npv)
    n_valid = int(valid.sum())
    failed = n_valid < int(e["min_valid"])
    samples = np.array([float(e["failure_npv_bn"])]) if failed else npv[valid]
    return {
        "problem": problem.key, "archetype": problem.archetype.__dict__, "vector": problem.vector,
        "trains": problem.trains, "train_size": problem.train_size,
        "x": [float(v) for v in x], "x_keys": SEARCH_KEYS, "design": design,
        "seeds": seeds(cfg), "n_valid": n_valid, "failed": bool(failed),
        "samples_bn": [float(v) for v in samples],
        "mean_npv_bn": float(np.mean(npv[valid])) if n_valid else float("nan"),
        "sd_npv_bn": float(np.std(npv[valid], ddof=1)) if n_valid > 1 else float("nan"),
        "per_seed": [{k: r.get(k) for k in DIAGNOSTICS + ("seed", "error")} for r in runs],
        "eval_wall_s": time.time() - t0, "ncpus": ncpus,
    }


def write_json(path: Path, data: dict) -> None:
    """Atomic write (tmp + rename), so a killed job never leaves a half-written result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1, default=float)
    os.replace(tmp, path)


def load_results(dirs: list[Path], problem_key: str) -> list[dict]:
    """Every finished result file for a problem across `dirs` (Sobol and BO), in file-name order."""
    out = []
    for d in dirs:
        for p in sorted(Path(d).glob(f"{problem_key}/*.json")):
            try:
                r = json.loads(p.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if r.get("problem") == problem_key and "samples_bn" in r:
                out.append(r)
    return out


def ncpus_default() -> int:
    for var in ("NCPUS", "PBS_NCPUS", "SLURM_CPUS_PER_TASK"):
        if os.environ.get(var):
            return int(os.environ[var])
    return os.cpu_count() or 1
