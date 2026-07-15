"""Acquisition maximisation: Sobol screen + multistart SQP on the direct
driver (`run_sqp`). Integer combos are enumerated and vmapped below the
restart axis; combo values reach the objective by closure.
"""

from __future__ import annotations

from itertools import product
from typing import Optional, Sequence, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from scipy.stats.qmc import Sobol

from septal.jax.sqp import SQPConfig

from hydro_bo.opt.acquisition import AcquisitionFunction
from hydro_bo.opt.sqp_driver import run_sqp
from hydro_bo.utils.logging_config import get_logger

logger = get_logger(__name__)


DEFAULT_SQP_CONFIG = SQPConfig(
    max_iter=300,
    use_exact_hessian=True,
    tol_stationarity=1e-6,
    tol_feasibility=1e-6,
)


def sobol_sample(bounds: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Quasi-random points in `bounds` for the BO initial-design phase."""
    sampler = Sobol(d=bounds.shape[0], scramble=True, seed=seed)
    return bounds[:, 0] + sampler.random(n) * (bounds[:, 1] - bounds[:, 0])


def clear_jax_caches() -> None:
    """Drop XLA-compiled artifacts (pad growth invalidates traces)."""
    try:
        jax.clear_caches()
    except AttributeError:
        try:
            from jax._src.compilation_cache import compilation_cache as _cc

            _cc.reset_cache()
        except Exception:
            pass


def _pmap_batch_eval(batch_fn, x_batch: jnp.ndarray, *args) -> jnp.ndarray:
    """Shard `x_batch`'s leading axis across devices via pmap; fall back
    to the plain vmapped call when only one device is available or the
    batch isn't divisible."""
    n_dev = jax.local_device_count()
    n_total = int(x_batch.shape[0])
    if n_dev <= 1 or n_total % n_dev != 0:
        return batch_fn(x_batch, *args)
    per_dev = n_total // n_dev
    x_sharded = x_batch.reshape((n_dev, per_dev) + x_batch.shape[1:])
    pmap_fn = jax.pmap(batch_fn, in_axes=(0,) + (None,) * len(args))
    out = pmap_fn(x_sharded, *args)
    return out.reshape((n_total,) + out.shape[2:])


def _insert(x_cont, ints, idx: tuple):
    """Insert `ints` into `x_cont` at static positions `idx` (ascending)."""
    x = x_cont
    for i, k in enumerate(idx):
        fill = jnp.broadcast_to(ints[i].astype(x.dtype), x.shape[:-1] + (1,))
        x = jnp.concatenate([x[..., :k], fill, x[..., k:]], axis=-1)
    return x


def _insert_np(x_cont: np.ndarray, ints: np.ndarray, idx: tuple) -> np.ndarray:
    x = np.asarray(x_cont, dtype=float)
    for i, k in enumerate(idx):
        x = np.insert(x, k, float(ints[i]))
    return x


class NLPBase:
    """Box-constrained multi-start SQP over the unit cube."""

    def __init__(
        self,
        seed: int = 0,
        pow_sobol: int = 14,
        n_restarts: int = 5,
        sqp_config: Optional[SQPConfig] = None,
    ):
        self.seed = int(seed)
        self.pow_sobol = int(pow_sobol)
        self.n_restarts = int(n_restarts)
        self.sqp_config = sqp_config or DEFAULT_SQP_CONFIG

    def maximise(self, acq: AcquisitionFunction) -> tuple[np.ndarray, float]:
        import time as _time

        D = self._dim(acq)
        cat_idx, combos = self._layout()
        D_red = D - len(cat_idx)
        combos_np = np.asarray(combos, dtype=float).reshape(len(combos), len(cat_idx))
        C, R = combos_np.shape[0], self.n_restarts

        t_screen = _time.perf_counter()
        starts, screen_scores = self._screen(acq, D_red, cat_idx, combos_np)
        screen_seconds = _time.perf_counter() - t_screen

        t_sqp = _time.perf_counter()
        states = self._solve(acq, cat_idx, starts, combos_np, D_red)
        sqp_seconds = _time.perf_counter() - t_sqp

        conv = np.asarray(states.converged).reshape(C, R)
        f_val = np.asarray(states.f_val).reshape(C, R)
        xs = np.asarray(states.x).reshape(C, R, D_red)

        logger.info(
            "acq_optimise_timing",
            solver=type(self).__name__,
            n_starts=C * R,
            n_combos=C,
            sobol_pow=self.pow_sobol,
            screen_seconds=float(screen_seconds),
            sqp_seconds=float(sqp_seconds),
            n_converged=int(conv.sum()),
        )

        if not conv.any():
            c, r = np.unravel_index(int(np.argmax(screen_scores)), (C, R))
            x_full = _insert_np(starts[c, r], combos_np[c], cat_idx)
            logger.warning(
                "sqp_no_converged_starts",
                n_starts=C * R,
                fallback=float(screen_scores[c, r]),
            )
            return np.clip(x_full, 0.0, 1.0), float(screen_scores[c, r])

        masked = np.where(conv, f_val, np.inf)
        c, r = np.unravel_index(int(np.argmin(masked)), (C, R))
        x_full = _insert_np(xs[c, r], combos_np[c], cat_idx)
        return np.clip(x_full, 0.0, 1.0), float(-masked[c, r])

    # ---- subclass hooks ----

    def _layout(self) -> tuple[tuple[int, ...], list[tuple[float, ...]]]:
        return (), [()]

    def _constraint(self, acq: AcquisitionFunction, cat_idx: tuple):
        """Returns (con_fn(x_full) -> (n_g,), lhs, rhs) or None."""
        return None

    def _rescore(self, acq: AcquisitionFunction, raw: np.ndarray, x_full) -> np.ndarray:
        return raw

    # ---- shared infrastructure ----

    @staticmethod
    def _dim(acq: AcquisitionFunction) -> int:
        return int(acq.state_args[0]["X"].shape[1])

    def _screen(self, acq, D_red, cat_idx, combos_np):
        """Sobol candidates in the continuous dims, assembled to full
        inputs per combo, screened by the batched acquisition. Returns
        (starts (C, R, D_red), scores (C, R))."""
        sampler = Sobol(d=D_red, scramble=True, seed=self.seed)
        candidates = sampler.random(2**self.pow_sobol)
        cand_j = jnp.asarray(candidates, dtype=jnp.float64)
        batch_fn = acq.acq_batch_fn()
        state = acq.state_args

        starts, scores = [], []
        for combo in combos_np:
            ints = jnp.asarray(combo, dtype=jnp.float64)
            x_full = _insert(cand_j, ints, cat_idx)
            raw = np.asarray(_pmap_batch_eval(batch_fn, x_full, *state))
            adj = self._rescore(acq, raw, x_full)
            top = np.argsort(adj)[-self.n_restarts :]
            starts.append(candidates[top])
            scores.append(adj[top])
        return np.stack(starts), np.stack(scores)

    def _solve(self, acq, cat_idx, starts, combos_np, D_red):
        neg = acq.neg_acq_fn()
        state = acq.state_args
        con = self._constraint(acq, cat_idx)
        lb = jnp.zeros(D_red, dtype=jnp.float64)
        ub = jnp.ones(D_red, dtype=jnp.float64)
        cfg = self.sqp_config

        def solve_cell(x0, ints):
            neg_obj = lambda xc: neg(_insert(xc, ints, cat_idx), *state).reshape(())
            if con is None:
                return run_sqp(neg_obj, x0, lb, ub, cfg)
            con_fn, lhs, rhs = con
            g = lambda xc: con_fn(_insert(xc, ints, cat_idx))
            return run_sqp(neg_obj, x0, lb, ub, cfg, con=g, con_lhs=lhs, con_rhs=rhs)

        per_restart = jax.vmap(solve_cell, in_axes=(0, None))
        per_combo = jax.vmap(per_restart, in_axes=(0, 0))
        starts_j = jnp.asarray(starts, dtype=jnp.float64)
        combos_j = jnp.asarray(combos_np, dtype=jnp.float64)
        return jax.jit(per_combo)(starts_j, combos_j)


class MixedIntNLP(NLPBase):
    """+ integer branching: one combo per element of the cartesian
    product of levels, vmapped below the restart axis."""

    def __init__(
        self,
        cat_vars: Sequence[Tuple[int, Sequence[float]]] = (),
        seed: int = 0,
        pow_sobol: int = 14,
        n_restarts: int = 5,
        sqp_config: Optional[SQPConfig] = None,
    ):
        super().__init__(
            seed=seed, pow_sobol=pow_sobol, n_restarts=n_restarts, sqp_config=sqp_config
        )
        self.cat_vars = [(int(i), [float(v) for v in vals]) for i, vals in cat_vars]

    def _layout(self):
        if not self.cat_vars:
            return (), [()]
        idxs, vals = zip(*self.cat_vars)
        order = sorted(range(len(idxs)), key=lambda i: idxs[i])
        cat = tuple(int(idxs[i]) for i in order)
        return cat, list(product(*[list(vals[i]) for i in order]))


class ConstrainedMixedIntNLP(MixedIntNLP):
    """+ chance-bound feasibility constraint inside the SQP and an L1
    hinge re-ranking the Sobol screen. Both operate on the latent-space
    LHS from the acquisition (`f(x) + z_sc·σ(x) − log_p_targ`);
    feasibility means LHS ≥ 0."""

    def __init__(
        self,
        cat_vars: Sequence[Tuple[int, Sequence[float]]] = (),
        l1_penalty: float = 1.0,
        seed: int = 0,
        pow_sobol: int = 14,
        n_restarts: int = 5,
        sqp_config: Optional[SQPConfig] = None,
    ):
        super().__init__(
            cat_vars=cat_vars,
            seed=seed,
            pow_sobol=pow_sobol,
            n_restarts=n_restarts,
            sqp_config=sqp_config,
        )
        self.l1_penalty = float(l1_penalty)

    def _constraint(self, acq, cat_idx):
        feas = acq.feasibility_eval_fn()
        feas_state = acq.feasibility_state_args
        con_fn = lambda x_full: jnp.atleast_1d(feas(x_full, *feas_state))
        lhs = jnp.array([0.0], dtype=jnp.float64)
        rhs = jnp.array([jnp.inf], dtype=jnp.float64)
        return con_fn, lhs, rhs

    def _rescore(self, acq, raw, x_full):
        if self.l1_penalty <= 0.0:
            return raw
        feas_fn = acq.feasibility_batch_fn()
        feas_state = acq.feasibility_state_args
        lhs = np.asarray(_pmap_batch_eval(feas_fn, x_full, *feas_state))
        return raw - self.l1_penalty * np.maximum(0.0, -lhs)
