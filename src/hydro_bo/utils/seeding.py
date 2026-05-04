"""
Centralised seed handling.

All randomness in this codebase flows through one of three helpers:

    resolve_master_seed(cli_seed)      # entry-point: scripts pick a master seed
    derive_worker_seed(master, ...)    # dispatcher: per-Ray-task seed
    derive_subseed(parent, label)      # consumers: per-component sub-streams
    make_rng(seed)                     # any consumer: Generator from a seed

Design choices:
  * Sub-seeds are 64-bit, derived via blake2s — no entropy lost to truncation.
  * Worker seeds mix master_seed + PBS_JOBID + PBS_ARRAY_INDEX + pid + time_ns
    + ray_task_id, so two PBS array elements never share a worker seed even
    when Ray restarts and the per-task counter resets.
  * Sub-seeds for distinct components are deterministic from (parent, label)
    — so weather, ship-arrival noise, price dynamics, and gurobi each get
    their own independent stream from the same master, reproducibly.
  * No code outside this module should call np.random.seed or use the global
    np.random module.
"""

from __future__ import annotations

import hashlib
import os
import time
from typing import Optional

import numpy as np


_SEED_BYTES = 8  # 64-bit seeds


def _hash_to_int(*parts: object) -> int:
    """Stable 64-bit hash of the concatenated string forms of `parts`."""
    h = hashlib.blake2s(digest_size=_SEED_BYTES)
    for p in parts:
        h.update(repr(p).encode("utf-8"))
        h.update(b"\x00")
    return int.from_bytes(h.digest(), "big")


def resolve_master_seed(cli_seed: Optional[int]) -> int:
    """Return the master seed for this run.

    If ``cli_seed`` is provided, it is mixed with PBS_ARRAY_INDEX so that a
    single CLI invocation across an array job still yields per-element-unique
    masters. If ``cli_seed`` is None, derive a fresh seed from PBS env vars,
    pid, and the wall clock — so two concurrent runs without an explicit seed
    will not collide.
    """
    pbs_jobid = os.environ.get("PBS_JOBID", "")
    pbs_array = os.environ.get("PBS_ARRAY_INDEX", "")

    if cli_seed is not None:
        return _hash_to_int("master", int(cli_seed), pbs_jobid, pbs_array)

    return _hash_to_int(
        "master",
        pbs_jobid,
        pbs_array,
        os.getpid(),
        time.time_ns(),
    )


def derive_worker_seed(master_seed: int, ray_task_id: str) -> int:
    """Per-Ray-task seed.

    Mixes master_seed with everything that distinguishes one worker invocation
    from another, including across PBS array elements and Ray restarts.
    """
    return _hash_to_int(
        "worker",
        int(master_seed),
        os.environ.get("PBS_JOBID", ""),
        os.environ.get("PBS_ARRAY_INDEX", ""),
        os.getpid(),
        time.time_ns(),
        ray_task_id,
    )


def derive_subseed(parent_seed: int, label: str) -> int:
    """Deterministic sub-seed for a named component (weather, price, ...)."""
    return _hash_to_int("subseed", int(parent_seed), label)


def make_rng(seed: int) -> np.random.Generator:
    """numpy Generator from a 64-bit seed (no global state mutation)."""
    return np.random.default_rng(int(seed) & ((1 << 64) - 1))


def gurobi_seed(seed: int) -> int:
    """Clamp a seed into Gurobi's accepted range (0..2_000_000_000)."""
    return int(seed) % 2_000_000_000
