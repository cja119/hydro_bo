"""NLP solvers for acquisition-function optimisation, septal-backed.

  - `NLPBase`: box-constrained multi-start SQP over the unit cube.
  - `MixedIntNLP`: + categorical / integer branching via septal's
    parametric `p` (one factory; `solve_batch` runs every combo in one
    dispatch).
  - `ConstrainedMixedIntNLP`: + explicit feasibility constraint
    `p_targ <= P(feasible|x) <= 1` and an L1 hinge in the Sobol screen.

Polymorphism is on `_solver(acq) -> ParametricNLPProblem` and
`_build_starts(acq) -> (starts, p_batch, screen_scores)`. The shared
`_build_solver` and `maximise` live once on the base.
"""

from __future__ import annotations

import dataclasses
from itertools import product
from typing import Optional, Sequence, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from scipy.stats.qmc import Sobol

from septal.jax.sqp import (
    ParametricNLPProblem,
    ParametricSQPFactory,
    SQPConfig,
)

from hydro_bo.opt.acquisition import AcquisitionFunction
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
    """Drop XLA-compiled artifacts (each BO iter triggers a fresh trace)."""
    try:
        jax.clear_caches()
    except AttributeError:
        try:
            from jax._src.compilation_cache import compilation_cache as _cc

            _cc.reset_cache()
        except Exception:
            pass


def _pmap_batch_eval(batch_fn, x_batch: jnp.ndarray, *args) -> jnp.ndarray:
    """Shard `x_batch`'s leading axis across `jax.local_device_count()`
    logical devices via pmap; fall back to the plain (vmap-fused) call
    when only one device is available or the batch isn't divisible.

    `batch_fn(x, *args)` is expected to be a vmapped callable mapping
    `(B, ...) -> (B,)`. The trailing args are broadcast (not sharded).
    Static args (e.g. `round_info`) should be bound upstream via
    `functools.partial` so we don't need to thread them through pmap.
    """
    n_dev = jax.local_device_count()
    n_total = int(x_batch.shape[0])
    if n_dev <= 1 or n_total % n_dev != 0:
        return batch_fn(x_batch, *args)
    per_dev = n_total // n_dev
    x_sharded = x_batch.reshape((n_dev, per_dev) + x_batch.shape[1:])
    pmap_fn = jax.pmap(batch_fn, in_axes=(0,) + (None,) * len(args))
    out = pmap_fn(x_sharded, *args)
    return out.reshape((n_total,) + out.shape[2:])


# ---------------------------------------------------------------------- #


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

    # ---- public ----

    def maximise(self, acq: AcquisitionFunction) -> tuple[np.ndarray, float]:
        import time as _time
        D = self._dim(acq)
        t_starts = _time.perf_counter()
        starts, p_batch, screen_scores = self._build_starts(acq)
        starts_seconds = _time.perf_counter() - t_starts
        n_starts = int(starts.shape[0])
        t_sqp = _time.perf_counter()
        result = self._build_solver(acq).solve_batch(starts, p_batch)
        sqp_seconds = _time.perf_counter() - t_sqp
        logger.info(
            "acq_optimise_timing",
            solver=type(self).__name__,
            n_starts=n_starts,
            n_combos=int(n_starts // max(self.n_restarts, 1)),
            sobol_pow=int(self.pow_sobol),
            screen_seconds=float(starts_seconds),
            sqp_seconds=float(sqp_seconds),
            sqp_inner_timing=float(getattr(result, "timing", float("nan"))),
            n_converged=int(np.asarray(result.success).astype(bool).sum()),
        )
        return self._select_best(result, starts, p_batch, screen_scores, D)

    # ---- subclass hooks ----

    def _build_starts(
        self,
        acq: AcquisitionFunction,
    ) -> tuple[jnp.ndarray, jnp.ndarray, np.ndarray]:
        """Returns (starts, p_batch, screen_scores). Base = single-combo
        Sobol screen on the full unit cube; subclasses override to
        enumerate categorical combos."""
        D = self._dim(acq)
        candidates, scores = self._sobol_screen(acq, D, acq.acq_batch_fn())
        return (
            jnp.asarray(candidates, dtype=jnp.float64),
            jnp.zeros((self.n_restarts, 0), dtype=jnp.float64),
            scores,
        )

    def _solver(self, acq: AcquisitionFunction) -> ParametricNLPProblem:
        D = self._dim(acq)
        return ParametricNLPProblem(
            objective=self._objective(acq, ()),
            bounds=[jnp.zeros(D, dtype=jnp.float64), jnp.ones(D, dtype=jnp.float64)],
            n_decision=D,
            n_params=0,
        )

    # ---- shared infrastructure ----

    def _build_solver(self, acq: AcquisitionFunction) -> ParametricSQPFactory:
        """Wrap `self._solver(acq)` into a septal factory. Never overridden."""
        return ParametricSQPFactory(self._solver(acq), self.sqp_config)

    @staticmethod
    def _dim(acq: AcquisitionFunction) -> int:
        return int(acq.state_args[0]["X"].shape[1])

    @staticmethod
    def _objective(acq: AcquisitionFunction, cat_indices: tuple[int, ...]):
        """Wrap acq into septal's `obj(x, p) -> scalar` (negated for SQP min)."""
        state = acq.state_args
        if cat_indices:
            neg = acq.neg_acq_masked_fn(cat_indices)
            return lambda x, p: neg(x, p, *state).reshape(())
        neg = acq.neg_acq_fn()
        return lambda x, p: neg(x, *state).reshape(())

    def _sobol_screen(
        self,
        acq: AcquisitionFunction,
        dim: int,
        batch_fn,
        extra_args: tuple = (),
        rescore=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """2**pow_sobol candidates → batch eval → top-K. Optional `rescore`
        callback `(raw_scores, x_jnp) -> adjusted_scores` lets subclasses
        re-rank without a full override."""
        sampler = Sobol(d=dim, scramble=True, seed=self.seed)
        candidates = sampler.random(2**self.pow_sobol)
        x_jnp = jnp.asarray(candidates, dtype=jnp.float64)
        raw = np.asarray(
            _pmap_batch_eval(batch_fn, x_jnp, *extra_args, *acq.state_args)
        )
        scores = rescore(raw, x_jnp) if rescore is not None else raw
        top_idx = np.argsort(scores)[-self.n_restarts :]
        return candidates[top_idx], scores[top_idx]

    def _select_best(
        self,
        result,
        starts,
        p_batch,
        screen_scores,
        dim_full,
    ) -> tuple[np.ndarray, float]:
        """Best converged start (SQP enforces feasibility internally), with
        screen-score fallback if no start converged."""
        success = np.asarray(result.success).astype(bool)
        objectives = np.asarray(result.objective)

        if not np.any(success):
            best = int(np.argmax(screen_scores))
            x_full = self._expand_x(
                np.asarray(starts)[best],
                np.asarray(p_batch)[best],
                dim_full,
            )
            logger.warning(
                "sqp_no_converged_starts",
                n_starts=int(success.size),
                fallback=float(screen_scores[best]),
            )
            return x_full, float(screen_scores[best])

        best = int(np.argmin(np.where(success, objectives, np.inf)))
        x_full = self._expand_x(
            np.asarray(result.decision_variables)[best],
            np.asarray(p_batch)[best],
            dim_full,
        )
        return x_full, float(-objectives[best])

    def _expand_x(self, x_red, p, dim_full) -> np.ndarray:
        """Reconstruct the full unit-cube x. Base case: x_red IS x_full."""
        return np.clip(np.asarray(x_red, dtype=float), 0.0, 1.0)


# ---------------------------------------------------------------------- #


class MixedIntNLP(NLPBase):
    """+ categorical/integer branching. Empty cat_vars short-circuits
    to the continuous base path."""

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

    # ---- subclass hooks (inherited maximise drives these) ----

    def _build_starts(self, acq):
        if not self.cat_vars:
            return super()._build_starts(acq)
        D = self._dim(acq)
        cat, combos = self._layout()
        D_red = D - len(cat)
        batch_fn = acq.acq_batch_masked_fn(cat)

        starts, ps, scores = [], [], []
        for combo in combos:
            mv = jnp.asarray(combo, dtype=jnp.float64)
            cand, sc = self._sobol_screen(acq, D_red, batch_fn, extra_args=(mv,))
            starts.append(np.asarray(cand))
            ps.append(
                np.broadcast_to(
                    np.asarray(combo, dtype=float),
                    (self.n_restarts, len(cat)),
                ).copy()
            )
            scores.append(np.asarray(sc))

        return (
            jnp.asarray(np.concatenate(starts, 0), dtype=jnp.float64),
            jnp.asarray(np.concatenate(ps, 0), dtype=jnp.float64),
            np.concatenate(scores, 0),
        )

    def _solver(self, acq):
        if not self.cat_vars:
            return super()._solver(acq)
        D = self._dim(acq)
        cat, _ = self._layout()
        D_red = D - len(cat)
        return ParametricNLPProblem(
            objective=self._objective(acq, cat),
            bounds=[
                jnp.zeros(D_red, dtype=jnp.float64),
                jnp.ones(D_red, dtype=jnp.float64),
            ],
            n_decision=D_red,
            n_params=len(cat),
        )

    def _expand_x(self, x_red, p, dim_full):
        if not self.cat_vars:
            return super()._expand_x(x_red, p, dim_full)
        cat, _ = self._layout()
        x_red = np.asarray(x_red, dtype=float)
        p = np.asarray(p, dtype=float)
        x_full = np.empty(dim_full, dtype=float)
        cat_set = set(cat)
        cont = [i for i in range(dim_full) if i not in cat_set]
        for i, idx in enumerate(cat):
            x_full[idx] = p[i]
        x_full[cont] = x_red
        return np.clip(x_full, 0.0, 1.0)

    # ---- helpers ----

    def _layout(self) -> tuple[tuple[int, ...], list[tuple[float, ...]]]:
        """Sorted cat indices + cartesian product of mask values."""
        if not self.cat_vars:
            return (), [()]
        idxs, vals = zip(*self.cat_vars)
        order = sorted(range(len(idxs)), key=lambda i: idxs[i])
        cat = tuple(int(idxs[i]) for i in order)
        return cat, list(product(*[list(vals[i]) for i in order]))


# ---------------------------------------------------------------------- #


class ConstrainedMixedIntNLP(MixedIntNLP):
    """+ explicit chance-bound feasibility constraint inside the SQP,
    and an L1 hinge re-ranking the Sobol screen.

    Both mechanisms operate on the latent-space LHS supplied by the
    acquisition (`f(x) + z_sc·σ(x) − log_p_targ`); feasibility means
    `LHS ≥ 0`. The threshold (i.e. `p_targ`) is baked into the
    acquisition's `log_p_targ`, so this class only carries `l1_penalty`.

    SQP constraint: `0 ≤ LHS ≤ +∞`.
    L1 hinge: `raw − l1_penalty · max(0, −LHS)` — no penalty for
    feasible points, linear penalty proportional to the chance-bound
    shortfall otherwise.
    """

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

    def _solver(self, acq):
        parent = super()._solver(acq)
        cat, _ = self._layout()
        feas_state = acq.feasibility_state_args
        if cat:
            feas = acq.feasibility_eval_masked_fn(cat)
            g = lambda x, p: jnp.atleast_1d(feas(x, p, *feas_state))
        else:
            feas = acq.feasibility_eval_fn()
            g = lambda x, p: jnp.atleast_1d(feas(x, *feas_state))
        return dataclasses.replace(
            parent,
            constraints=g,
            constraint_lhs=jnp.array([0.0], dtype=jnp.float64),
            constraint_rhs=jnp.array([jnp.inf], dtype=jnp.float64),
            n_constraints=1,
        )

    def _sobol_screen(self, acq, dim, batch_fn, extra_args=(), rescore=None):
        if rescore is None and self.l1_penalty > 0.0:
            rescore = self._l1_rescore(acq, extra_args)
        return super()._sobol_screen(acq, dim, batch_fn, extra_args, rescore=rescore)

    def _l1_rescore(self, acq, extra_args):
        """Build the L1-hinge rescore callback: subtracts
        `l1_penalty * max(0, -LHS)` from the raw acquisition values,
        where LHS is the latent-space chance-bound LHS."""
        cat, _ = self._layout()
        feas_state = acq.feasibility_state_args
        if cat:
            feas_fn = acq.feasibility_batch_masked_fn(cat)
            mask_values = extra_args[0]

            def rescore(raw, x_jnp):
                feas = np.asarray(
                    _pmap_batch_eval(feas_fn, x_jnp, mask_values, *feas_state)
                )
                return raw - self.l1_penalty * np.maximum(0.0, -feas)

        else:
            feas_fn = acq.feasibility_batch_fn()

            def rescore(raw, x_jnp):
                feas = np.asarray(
                    _pmap_batch_eval(feas_fn, x_jnp, *feas_state)
                )
                return raw - self.l1_penalty * np.maximum(0.0, -feas)

        return rescore


def multistart_sqp(
    objective_fn,
    bounds: Tuple[np.ndarray, np.ndarray],
    *,
    seed: int,
    pow_sobol: int = 10,
    n_restarts: int = 8,
    sqp_config: Optional[SQPConfig] = None,
    n_params: int = 0,
    constraints: Optional[dict] = None,
) -> tuple[np.ndarray, float, dict]:
    """Sobol-screen + multistart septal SQP for any box-constrained NLP.

    The SQP minimises `objective_fn(x, p)`. Sobol-screens 2**pow_sobol
    candidates over the box, takes the `n_restarts` lowest-objective
    starts, vmaps the SQP solve across them, and returns the best
    converged result. Falls back to the best Sobol candidate if no
    start converges.
    """
    cfg = sqp_config or DEFAULT_SQP_CONFIG
    lb_np = np.asarray(bounds[0], dtype=float)
    ub_np = np.asarray(bounds[1], dtype=float)
    d = int(lb_np.shape[0])
    lb = jnp.asarray(lb_np, dtype=jnp.float64)
    ub = jnp.asarray(ub_np, dtype=jnp.float64)

    # Sobol cloud over the box, screen by raw objective (lower is better).
    sampler = Sobol(d=d, scramble=True, seed=seed)
    candidates_unit = sampler.random(2**pow_sobol)
    candidates = lb_np + candidates_unit * (ub_np - lb_np)
    x_jnp = jnp.asarray(candidates, dtype=jnp.float64)
    p_zero = jnp.zeros(n_params, dtype=jnp.float64)
    obj_batch = jax.vmap(objective_fn, in_axes=(0, None))(x_jnp, p_zero)
    obj_np = np.asarray(obj_batch)

    top_idx = np.argsort(obj_np)[:n_restarts]
    starts = candidates[top_idx]
    starts_jnp = jnp.asarray(starts, dtype=jnp.float64)
    p_batch = jnp.zeros((n_restarts, n_params), dtype=jnp.float64)

    # Build problem (with optional inequality constraint).
    problem_kwargs = dict(
        objective=objective_fn,
        bounds=[lb, ub],
        n_decision=d,
        n_params=n_params,
    )
    if constraints is not None:
        lhs = jnp.asarray(constraints["lhs"], dtype=jnp.float64).reshape(-1)
        rhs = jnp.asarray(constraints["rhs"], dtype=jnp.float64).reshape(-1)
        problem_kwargs.update(
            constraints=constraints["fn"],
            constraint_lhs=lhs,
            constraint_rhs=rhs,
            n_constraints=int(lhs.shape[0]),
        )
    factory = ParametricSQPFactory(ParametricNLPProblem(**problem_kwargs), cfg)
    result = factory.solve_batch(starts_jnp, p_batch)

    success = np.asarray(result.success).astype(bool)
    objectives = np.asarray(result.objective)
    decisions = np.asarray(result.decision_variables)

    if not np.any(success):
        best = int(np.argmin(obj_np[top_idx]))
        x_best = np.clip(starts[best], lb_np, ub_np)
        val_best = float(obj_np[top_idx][best])
        info = {
            "converged": False,
            "n_converged": 0,
            "n_starts": int(n_restarts),
            "best_screen_obj": val_best,
            "fallback": "sobol",
        }
        logger.warning(
            "multistart_sqp_no_converged_starts",
            n_starts=int(n_restarts),
            fallback_obj=val_best,
        )
        return x_best, val_best, info

    masked = np.where(success, objectives, np.inf)
    best = int(np.argmin(masked))
    x_best = np.clip(decisions[best], lb_np, ub_np)
    val_best = float(objectives[best])
    info = {
        "converged": True,
        "n_converged": int(np.sum(success)),
        "n_starts": int(n_restarts),
        "best_screen_obj": float(np.min(obj_np[top_idx])),
    }
    return x_best, val_best, info
