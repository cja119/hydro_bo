"""Acquisition maximisation: Sobol screen + multistart SQP on the direct
driver (`run_sqp`). Integer combos are enumerated and vmapped below the
restart axis; combo values reach the objective by closure.
"""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from scipy.stats.qmc import Sobol

from septal.jax.sqp import SQPConfig


from hydro_bo.opt.sqp_driver import run_sqp
from hydro_bo.utils.logging_config import get_logger

logger = get_logger(__name__)


DEFAULT_SQP_CONFIG = SQPConfig(
    max_iter=300,
    use_exact_hessian=True,
    tol_stationarity=1e-6,
    tol_feasibility=1e-6,
)

if TYPE_CHECKING:
    from hydro_bo.opt.acquisition import AcquisitionFunction
    
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
        screen_chunk: Optional[int] = None,
    ):
        self.seed = int(seed)
        self.pow_sobol = int(pow_sobol)
        self.n_restarts = int(n_restarts)
        self.sqp_config = sqp_config or DEFAULT_SQP_CONFIG
        # Evaluate the Sobol screen in chunks of this many candidates so peak
        # memory (both the compiled batch graph and its run) is bounded
        # regardless of `2**pow_sobol`. None => one shot (fine for cheap EI;
        # KG sets it because each candidate eval is heavy — inner MIP solves
        # over theta x z — so a full-width batch OOMs).
        self.screen_chunk = int(screen_chunk) if screen_chunk else None

    def maximise(self, acq: 'AcquisitionFunction') -> tuple[np.ndarray, float]:
        import time as _time

        D = self._dim(acq)
        cat_idx, combos = self._layout()
        D_red = D - len(cat_idx)
        combos_np = np.asarray(combos, dtype=float).reshape(len(combos), len(cat_idx))
        C, R = combos_np.shape[0], self.n_restarts

        # Stage logging: the screen and the SQP polish can each take minutes
        # (compile + run), and previously nothing was logged between
        # `kg_acquisition_built` and the final `acq_optimise_timing`, so a long
        # acquisition looked like a hang. These bracket each stage so you can
        # see which one you're in and how long it took.
        n_cand = C * (2 ** self.pow_sobol)
        logger.info(
            "acq_screen_start", solver=type(self).__name__,
            n_candidates=n_cand, n_combos=C, sobol_pow=self.pow_sobol,
            screen_chunk=self.screen_chunk,
        )
        t_screen = _time.perf_counter()
        starts, screen_scores = self._screen(acq, D_red, cat_idx, combos_np)
        screen_seconds = _time.perf_counter() - t_screen
        logger.info(
            "acq_screen_done", screen_seconds=round(float(screen_seconds), 1),
        )

        logger.info("acq_solve_start", n_starts=C * R)
        t_sqp = _time.perf_counter()
        states = self._solve(acq, cat_idx, starts, combos_np, D_red)
        sqp_seconds = _time.perf_counter() - t_sqp
        logger.info(
            "acq_solve_done", sqp_seconds=round(float(sqp_seconds), 1),
        )

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

    def _constraint(self, acq: 'AcquisitionFunction', cat_idx: tuple):
        """Returns (con_fn(x_full) -> (n_g,), lhs, rhs) or None."""
        return None

    def _rescore(self, acq: 'AcquisitionFunction', raw: np.ndarray, x_full) -> np.ndarray:
        return raw

    # ---- shared infrastructure ----

    @staticmethod
    def _dim(acq: 'AcquisitionFunction') -> int:
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
        chunk = self.screen_chunk

        starts, scores = [], []
        for combo in combos_np:
            ints = jnp.asarray(combo, dtype=jnp.float64)
            x_full = _insert(cand_j, ints, cat_idx)
            if chunk and int(x_full.shape[0]) > chunk:
                # Bounded-memory screen: evaluate in fixed-size slices and
                # stitch. batch_fn compiles once for the `chunk` shape and is
                # reused for every full slice (a smaller trailing slice adds at
                # most one extra compile).
                parts = [
                    np.asarray(_pmap_batch_eval(batch_fn, x_full[i:i + chunk], *state))
                    for i in range(0, int(x_full.shape[0]), chunk)
                ]
                raw = np.concatenate(parts)
            else:
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
        screen_chunk: Optional[int] = None,
    ):
        super().__init__(
            seed=seed, pow_sobol=pow_sobol, n_restarts=n_restarts,
            sqp_config=sqp_config, screen_chunk=screen_chunk,
        )
        self.cat_vars = [(int(i), [float(v) for v in vals]) for i, vals in cat_vars]

    def _layout(self):
        if not self.cat_vars:
            return (), [()]
        idxs, vals = zip(*self.cat_vars)
        order = sorted(range(len(idxs)), key=lambda i: idxs[i])
        cat = tuple(int(idxs[i]) for i in order)
        return cat, list(product(*[list(vals[i]) for i in order]))


# Module-level caches of compiled screen/solve callables for MemoizedMixedIntNLP.
# Keyed on the STABLE acquisition callable identity + structure, so every BO
# iteration reuses ONE compiled program (GP state is passed as an argument).
_MEMO_SOLVE_CACHE: dict = {}


def _sqp_digest(cfg):
    import dataclasses
    return dataclasses.astuple(cfg) if dataclasses.is_dataclass(cfg) else (id(cfg),)


class MemoizedMixedIntNLP(MixedIntNLP):
    """MixedIntNLP whose screen and polish-solve jits are compiled ONCE and
    reused across BO iterations, instead of a fresh `jax.jit(closure)` per
    iteration.

    This is the "compile at the top level, pass state to the reused program"
    pattern (as EI/surrogate already do for their module-level jits), applied to
    the outer SQP solve — where, for KG, that compile is minutes and dominates.

    Two preconditions, both satisfied by the KG path:
      * the acquisition returns STABLE `neg`/`batch` callables (see
        `KnowledgeGradient.neg_acq_fn` / `acq_batch_fn`), so the cache key
        (their `id`) is constant across iterations;
      * the BO loop does NOT call `jax.clear_caches()` each iteration
        (`KnowledgeGradientBayesopt._clear_caches_each_iter = False`), which
        would otherwise wipe the reused compile.

    Unconstrained only (KG). GP state rides in as the `state` argument.
    """

    def _screen(self, acq, D_red, cat_idx, combos_np):
        # The acquisition's `batch` is itself a stable jitted vmap; call it
        # directly (no fresh per-iteration pmap) so it compiles once per chunk
        # shape and is reused every iteration.
        sampler = Sobol(d=D_red, scramble=True, seed=self.seed)
        candidates = sampler.random(2 ** self.pow_sobol)
        cand_j = jnp.asarray(candidates, dtype=jnp.float64)
        batch_fn = acq.acq_batch_fn()
        state = acq.state_args
        chunk = self.screen_chunk

        starts, scores = [], []
        for combo in combos_np:
            ints = jnp.asarray(combo, dtype=jnp.float64)
            x_full = _insert(cand_j, ints, cat_idx)
            if chunk and int(x_full.shape[0]) > chunk:
                parts = [
                    np.asarray(batch_fn(x_full[i:i + chunk], *state))
                    for i in range(0, int(x_full.shape[0]), chunk)
                ]
                raw = np.concatenate(parts)
            else:
                raw = np.asarray(batch_fn(x_full, *state))
            adj = self._rescore(acq, raw, x_full)
            top = np.argsort(adj)[-self.n_restarts :]
            starts.append(candidates[top])
            scores.append(adj[top])
        return np.stack(starts), np.stack(scores)

    def _solve(self, acq, cat_idx, starts, combos_np, D_red):
        state = acq.state_args
        cfg = self.sqp_config
        C = int(combos_np.shape[0])

        # Key on the CONSTANT structure only. `cfg` never changes (closed over,
        # so it needn't be hashable) and `neg` is behaviourally identical every
        # iteration (forwards to the stable memoised value with state as args),
        # so the solve is built ONCE on the first miss and reused thereafter.
        key = (_sqp_digest(cfg), tuple(cat_idx), int(D_red), C, int(self.n_restarts))
        solve_jit = _MEMO_SOLVE_CACHE.get(key)
        if solve_jit is None:
            if self._constraint(acq, cat_idx) is not None:
                raise NotImplementedError(
                    "MemoizedMixedIntNLP is for unconstrained (KG) solves only."
                )
            neg = acq.neg_acq_fn()        # captured once; forwards to stable value
            lb = jnp.zeros(D_red, dtype=jnp.float64)
            ub = jnp.ones(D_red, dtype=jnp.float64)

            def per_combo_fn(starts_j, combos_j, state):
                def solve_cell(x0, ints):
                    neg_obj = lambda xc: neg(
                        _insert(xc, ints, cat_idx), *state
                    ).reshape(())
                    return run_sqp(neg_obj, x0, lb, ub, cfg)
                per_restart = jax.vmap(solve_cell, in_axes=(0, None))
                per_combo = jax.vmap(per_restart, in_axes=(0, 0))
                return per_combo(starts_j, combos_j)

            solve_jit = jax.jit(per_combo_fn)
            _MEMO_SOLVE_CACHE[key] = solve_jit

        starts_j = jnp.asarray(starts, dtype=jnp.float64)
        combos_j = jnp.asarray(combos_np, dtype=jnp.float64)
        return solve_jit(starts_j, combos_j, state)


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
        screen_chunk: Optional[int] = None,
    ):
        super().__init__(
            cat_vars=cat_vars,
            seed=seed,
            pow_sobol=pow_sobol,
            n_restarts=n_restarts,
            sqp_config=sqp_config,
            screen_chunk=screen_chunk,
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


# ---------------------------------------------------------------------- #
# Traceable mixed-integer driver.
#
# Functional twin of `NLPBase.maximise` for use *inside* a trace — the
# class driver above is untouched and still serves EI/cEI. The difference
# is that scipy/numpy host calls are hoisted out (Sobol cloud and combo
# grid become literals built once), so the solve is pure jnp with static
# shapes and can be vmapped over an outer axis (KG's Hermite nodes).
# ---------------------------------------------------------------------- #


def sobol_cloud(dim: int, pow_sobol: int, seed: int = 0) -> jnp.ndarray:
    """Sobol screen cloud as a literal, built host-side once."""
    if int(dim) == 0:
        return jnp.zeros((1, 0), dtype=jnp.float64)
    sampler = Sobol(d=int(dim), scramble=True, seed=int(seed))
    return jnp.asarray(sampler.random(2 ** int(pow_sobol)), dtype=jnp.float64)


def combo_grid(cat_vars: Sequence[Tuple[int, Sequence[float]]]):
    """Array form of `MixedIntNLP._layout`: (cat_idx, combos (C, n_int))."""
    cat = [(int(i), [float(v) for v in vals]) for i, vals in cat_vars]
    if not cat:
        return (), jnp.zeros((1, 0), dtype=jnp.float64)
    order = sorted(range(len(cat)), key=lambda j: cat[j][0])
    cat_idx = tuple(cat[j][0] for j in order)
    combos = np.asarray(list(product(*[cat[j][1] for j in order])), dtype=float)
    return cat_idx, jnp.asarray(combos, dtype=jnp.float64)


def mip_solve(
    neg_obj,
    cloud: jnp.ndarray,
    combos: jnp.ndarray,
    cat_idx: tuple,
    lb: jnp.ndarray,
    ub: jnp.ndarray,
    cfg: SQPConfig,
    n_restarts: int = 1,
    warm_starts: Optional[jnp.ndarray] = None,
):
    """Traceable mixed-integer maximisation of `-neg_obj`.

    `neg_obj(x_full) -> scalar` is minimised, as everywhere else in the
    solver layer; the returned value is the corresponding maximum.
    Screens `cloud` per integer combo, takes the `n_restarts` best as SQP
    starts (prepending `warm_starts` when given), then solves every
    (combo, restart) cell with `run_sqp`.

    Returns (x_full_best (d,), value_best scalar).
    """
    C = int(combos.shape[0])
    d_red = int(lb.shape[0])

    def screen_combo(ints):
        return jax.vmap(lambda xr: -neg_obj(_insert(xr, ints, cat_idx)))(cloud)

    scores = jax.vmap(screen_combo)(combos)                    # (C, S)
    _, top = jax.lax.top_k(scores, int(n_restarts))            # (C, R_s)
    starts = jnp.take(cloud, top, axis=0)                      # (C, R_s, d_red)
    if warm_starts is not None:
        starts = jnp.concatenate([warm_starts, starts], axis=1)
    R = int(starts.shape[1])

    def solve_cell(x0, ints):
        return run_sqp(
            lambda xr: neg_obj(_insert(xr, ints, cat_idx)), x0, lb, ub, cfg,
        )

    per_restart = jax.vmap(solve_cell, in_axes=(0, None))
    per_combo = jax.vmap(per_restart, in_axes=(0, 0))
    st = per_combo(starts, combos)

    # Prefer converged cells; fall back to the raw best if none converged.
    raw = (-st.f_val).reshape(-1)
    masked = jnp.where(st.converged.reshape(-1), raw, -jnp.inf)
    any_conv = jnp.any(jnp.isfinite(masked))
    best = jnp.where(any_conv, jnp.argmax(masked), jnp.argmax(raw))
    val = raw[best]

    ci, ri = best // R, best % R
    x_red = st.x.reshape(C, R, d_red)[ci, ri]
    return _insert(x_red, combos[ci], cat_idx), val
