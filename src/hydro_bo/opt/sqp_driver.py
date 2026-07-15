"""Direct SQP driver over septal's array-level primitives.

The objective is a plain closure `neg_obj(x) -> scalar`; anything it is
conditioned on (integer combos, theta nodes, fantasy Z) is captured from
the enclosing scope, so there is no parameter vector `p`. The solve is
pure JAX (`lax.scan`, fixed `cfg.max_iter`, converged iterates frozen)
and composes under nested `jax.vmap`. Step arithmetic mirrors
`septal.jax.sqp.solver.make_sqp_step` one-for-one, pinned by
`tests/test_sqp_driver.py`.
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp

from septal.jax.sqp.schema import SQPConfig
from septal.jax.sqp.qp_subproblem import solve_qp_subproblem
from septal.jax.sqp.line_search import (
    backtracking_line_search,
    l1_merit,
    merit_directional_deriv,
    update_penalty,
)
from septal.jax.sqp.convergence import is_converged
from septal.jax.sqp.hessian import bfgs_update, regularised_lagrangian_hessian


class CellState(NamedTuple):
    """SQP iterate; a pytree of arrays, batchable under nested vmap."""

    x: jnp.ndarray
    lam: jnp.ndarray
    hessian: jnp.ndarray
    grad_lag: jnp.ndarray
    f_val: jnp.ndarray
    penalty: jnp.ndarray
    merit: jnp.ndarray
    stationarity: jnp.ndarray
    feasibility: jnp.ndarray
    converged: jnp.ndarray
    merit_window: jnp.ndarray
    stagnation_count: jnp.ndarray
    alpha_last: jnp.ndarray


def run_sqp(
    neg_obj: Callable[[jnp.ndarray], jnp.ndarray],
    x0: jnp.ndarray,
    lb: jnp.ndarray,
    ub: jnp.ndarray,
    cfg: SQPConfig,
    con: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    con_lhs: Optional[jnp.ndarray] = None,
    con_rhs: Optional[jnp.ndarray] = None,
) -> CellState:
    """Minimise `neg_obj(x)` s.t. `lb <= x <= ub` and, when `con` is
    given, `con_lhs <= con(x) <= con_rhs`. Array arguments may be
    traced; `cfg` is static. Runs exactly `cfg.max_iter` scan steps."""
    n = x0.shape[-1]
    n_g = int(con_lhs.shape[-1]) if con is not None else 0

    x0 = jnp.asarray(x0, dtype=jnp.float64).reshape(n)
    lb = jnp.asarray(lb, dtype=jnp.float64).reshape(n)
    ub = jnp.asarray(ub, dtype=jnp.float64).reshape(n)
    if n_g > 0:
        lhs = jnp.asarray(con_lhs, dtype=jnp.float64).reshape(n_g)
        rhs = jnp.asarray(con_rhs, dtype=jnp.float64).reshape(n_g)
    else:
        lhs = jnp.zeros(0, dtype=jnp.float64)
        rhs = jnp.zeros(0, dtype=jnp.float64)

    # septal's line search and exact-Hessian helpers expect f(x, p).
    p0 = jnp.zeros(0, dtype=jnp.float64)
    obj_p = lambda x, _p: jnp.asarray(neg_obj(x)).reshape(())
    con_p = (lambda x, _p: con(x).reshape(n_g)) if n_g > 0 else None

    def _grad_lag(x, lam):
        grad_f = jax.grad(lambda z: jnp.asarray(neg_obj(z)).reshape(()))(x)
        if n_g > 0:
            jac_g = jax.jacfwd(con)(x).reshape(n_g, n)
            return grad_f + jac_g.T @ lam
        return grad_f

    def _feas(g_val):
        if n_g == 0:
            return jnp.zeros((), dtype=jnp.float64)
        return jnp.maximum(
            jnp.max(jnp.maximum(g_val - rhs, 0.0)),
            jnp.max(jnp.maximum(lhs - g_val, 0.0)),
        )

    def _init() -> CellState:
        f0 = jnp.asarray(neg_obj(x0)).reshape(())
        lam0 = jnp.zeros(n_g, dtype=jnp.float64)
        g0 = con(x0).reshape(n_g) if n_g > 0 else jnp.zeros(0, dtype=jnp.float64)
        grad_f0 = jax.grad(lambda z: jnp.asarray(neg_obj(z)).reshape(()))(x0)
        penalty0 = jnp.array(cfg.penalty_init, dtype=jnp.float64)
        merit0 = l1_merit(f0, g0, lhs, rhs, penalty0, n_g)
        stat0 = jnp.max(jnp.abs(jnp.clip(x0 - grad_f0, lb, ub) - x0))
        feas0 = _feas(g0)
        return CellState(
            x=x0,
            lam=lam0,
            hessian=cfg.bfgs_init_scale * jnp.eye(n, dtype=jnp.float64),
            grad_lag=grad_f0,
            f_val=f0,
            penalty=penalty0,
            merit=merit0,
            stationarity=stat0,
            feasibility=feas0,
            converged=is_converged(stat0, feas0, cfg),
            merit_window=jnp.full(cfg.nonmonotone_window, merit0, dtype=jnp.float64),
            stagnation_count=jnp.array(0, dtype=jnp.int32),
            alpha_last=jnp.array(1.0, dtype=jnp.float64),
        )

    def _do_step(s: CellState) -> CellState:
        f_val, grad_f = jax.value_and_grad(
            lambda z: jnp.asarray(neg_obj(z)).reshape(())
        )(s.x)

        if n_g > 0:
            g_val = con(s.x).reshape(n_g)
            jac_g = jax.jacfwd(con)(s.x).reshape(n_g, n)
        else:
            g_val = jnp.zeros(0, dtype=jnp.float64)
            jac_g = jnp.zeros((0, n), dtype=jnp.float64)

        d, lam_new = solve_qp_subproblem(
            s.hessian, grad_f, jac_g, g_val, s.x, lb, ub, lhs, rhs, n, n_g, cfg,
            lam_prev=s.lam,
        )

        penalty_new = update_penalty(
            lam_new, s.penalty, n_g, cfg.penalty_eps,
            s.feasibility, cfg.penalty_decrease_factor,
        )

        reference_merit = jnp.max(s.merit_window)
        dir_deriv = merit_directional_deriv(
            grad_f, d, g_val, lhs, rhs, penalty_new, n_g
        )
        alpha = backtracking_line_search(
            s.x, d, p0,
            reference_merit, dir_deriv, penalty_new,
            obj_p, con_p,
            lhs, rhs, n_g, cfg,
        )

        x_new = s.x + alpha * d

        f_new = jnp.asarray(neg_obj(x_new)).reshape(())
        g_new = con(x_new).reshape(n_g) if n_g > 0 else jnp.zeros(0, dtype=jnp.float64)
        merit_new = l1_merit(f_new, g_new, lhs, rhs, penalty_new, n_g)

        grad_lag_new = _grad_lag(x_new, lam_new)

        if cfg.use_exact_hessian:
            H_new = regularised_lagrangian_hessian(
                x_new, lam_new, p0, obj_p, con_p, n_g,
                cfg.hess_reg_delta, cfg.hess_reg_min,
            )
        else:
            H_new = bfgs_update(
                s.hessian, x_new - s.x, grad_lag_new - s.grad_lag,
                cfg.bfgs_skip_tol, cfg.bfgs_max_cond,
            )

        stat_new = jnp.max(jnp.abs(jnp.clip(x_new - grad_lag_new, lb, ub) - x_new))
        feas_new = _feas(g_new)
        conv_new = is_converged(stat_new, feas_new, cfg)

        merit_window_new = jnp.roll(s.merit_window, -1).at[-1].set(merit_new)

        is_stagnating = alpha < cfg.stagnation_alpha_tol
        stagnation_new = jnp.where(
            is_stagnating, s.stagnation_count + 1, jnp.array(0, dtype=jnp.int32)
        )
        do_reset = stagnation_new >= cfg.stagnation_patience

        H_reset = cfg.bfgs_init_scale * jnp.eye(n, dtype=H_new.dtype)
        H_final = jnp.where(do_reset & cfg.stagnation_reset_hessian, H_reset, H_new)
        penalty_final = jnp.where(
            do_reset & cfg.stagnation_reset_penalty,
            jnp.array(cfg.penalty_init, dtype=jnp.float64),
            penalty_new,
        )
        stagnation_final = jnp.where(
            do_reset, jnp.array(0, dtype=jnp.int32), stagnation_new
        )

        return CellState(
            x=x_new,
            lam=lam_new,
            hessian=H_final,
            grad_lag=grad_lag_new,
            f_val=f_new,
            penalty=penalty_final,
            merit=merit_new,
            stationarity=stat_new,
            feasibility=feas_new,
            converged=conv_new,
            merit_window=merit_window_new,
            stagnation_count=stagnation_final,
            alpha_last=alpha,
        )

    def _step(s: CellState, _) -> tuple[CellState, None]:
        new = _do_step(s)
        frozen = jax.tree.map(
            lambda n_val, o_val: jnp.where(s.converged, o_val, n_val), new, s
        )
        return frozen._replace(converged=s.converged | new.converged), None

    final, _ = jax.lax.scan(_step, _init(), None, length=cfg.max_iter)
    return final


def best_cell(states: CellState, fallback_penalty: float = jnp.inf):
    """Reduce a batched `CellState` to the best converged cell:
    (x_best, f_best, flat_index)."""
    f_flat = jnp.where(
        states.converged, states.f_val, fallback_penalty
    ).reshape(-1)
    idx = jnp.argmin(f_flat)
    x_flat = states.x.reshape(-1, states.x.shape[-1])
    return x_flat[idx], f_flat[idx], idx
