"""
Utility functions for Bayesian optimization.
"""

from pathlib import Path
from typing import Optional

def ensure_dirs(tmp_dir: Optional[Path] = None, checkpoint_dir: Optional[Path] = None):
    base_tmp = (
        Path(tmp_dir)
        if tmp_dir is not None
        else Path(__file__).parent.parent.parent / "tmp"
    )
    bayesopt_dir = base_tmp / "bayesopt"
    ray_results_dir = base_tmp / "ray_results"
    ckpt_dir = (
        Path(checkpoint_dir) if checkpoint_dir is not None else base_tmp / "checkpoints"
    )
    for path in (bayesopt_dir, ray_results_dir, ckpt_dir):
        path.mkdir(parents=True, exist_ok=True)
    return base_tmp, bayesopt_dir, ray_results_dir, ckpt_dir

def parse_memory_string(memory_str: str) -> int:
    memory_str = memory_str.upper()
    if memory_str.endswith("GB"):
        return int(float(memory_str[:-2]) * 1024 * 1024 * 1024)
    if memory_str.endswith("MB"):
        return int(float(memory_str[:-2]) * 1024 * 1024)
    if memory_str.endswith("KB"):
        return int(float(memory_str[:-2]) * 1024)
    return int(memory_str)
