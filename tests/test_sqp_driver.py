"""Parity tests: `hydro_bo.opt.sqp_driver.run_sqp` vs septal's factory path.

The driver reimplements the SQP loop over septal's array-level
primitives with the objective as a closure (no parameter vector `p`).
These tests pin that the two paths produce identical iterates on
unconstrained and constrained problems, and that the closure pattern
composes under nested vmap with the integer axis below the theta axis.

Run directly (no pytest needed):  python tests/test_sqp_driver.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import jax
import jax.numpy as jnp

from septal.jax.sqp import (  # noqa: E402  (import enables x64)
    ParametricNLPProblem,
    ParametricSQPFactory,
    SQPConfig,
)
from hydro_bo.opt.sqp_driver import run_sqp, best_cell

CFG = SQPConfig(
    max_iter=60,
    use_exact_hessian=True,
    tol_stationarity=1e-8,
    tol_feasibility=1e-8,
)

D = 5
LB = jnp.zeros(D)
UB = jnp.ones(D)
_C = jnp.linspace(0.2, 0.8, D)
_W = jnp.asarray([1.0, 2.0, 3.0, 2.0, 1.0])


def _f(x):
    """Smooth non-convex objective on the unit box."""
    return jnp.sum(_W * (x - _C) ** 2) + 0.2 * jnp.sum(jnp.cos(4.0 * x))


def _starts(k, seed=0):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.uniform(0.05, 0.95, size=(k, D)))


def test_unconstrained_parity():
    problem = ParametricNLPProblem(
        objective=lambda x, p: _f(x), bounds=[LB, UB], n_decision=D, n_params=0,
    )
    factory = ParametricSQPFactory(problem, CFG)

    for x0 in _starts(4):
        ref = factory.solve(x0, jnp.zeros(0))
        got = run_sqp(_f, x0, LB, UB, CFG)
        assert bool(ref.success) == bool(got.converged)
        np.testing.assert_allclose(got.x, ref.decision_variables, atol=1e-10)
        np.testing.assert_allclose(got.f_val, ref.objective, atol=1e-10)
    print("test_unconstrained_parity PASSED")


def test_constrained_parity():
    c = 0.15 * jnp.ones(D)
    f = lambda x: jnp.sum((x - c) ** 2)
    g = lambda x: jnp.atleast_1d(jnp.sum(x) - 1.2)   # sum(x) >= 1.2, active at x*=c+
    lhs = jnp.array([0.0])
    rhs = jnp.array([jnp.inf])

    problem = ParametricNLPProblem(
        objective=lambda x, p: f(x),
        bounds=[LB, UB],
        n_decision=D,
        n_params=0,
        constraints=lambda x, p: g(x),
        constraint_lhs=lhs,
        constraint_rhs=rhs,
        n_constraints=1,
    )
    factory = ParametricSQPFactory(problem, CFG)

    x_analytic = c + (1.2 - float(jnp.sum(c))) / D   # projection onto sum(x)=1.2

    for x0 in _starts(4, seed=1):
        ref = factory.solve(x0, jnp.zeros(0))
        got = run_sqp(f, x0, LB, UB, CFG, con=g, con_lhs=lhs, con_rhs=rhs)
        assert bool(ref.success) == bool(got.converged)
        np.testing.assert_allclose(got.x, ref.decision_variables, atol=1e-10)
        np.testing.assert_allclose(got.f_val, ref.objective, atol=1e-10)
        assert bool(got.converged)
        np.testing.assert_allclose(got.x, x_analytic, atol=1e-6)
    print("test_constrained_parity PASSED")


def _center(ints, theta):
    """Quadratic-bowl centre conditioned on (integer combo, theta)."""
    ramp = 0.1 * jnp.arange(D) / D
    return 0.05 * ints + 0.2 * theta + ramp


def test_nested_vmap_closure():
    """Integer axis vmapped below the theta axis, parameters via closure.
    The argmin of a quadratic bowl is its centre, so every cell has an
    analytic solution."""

    def solve_cell(x0, ints, theta):
        neg_obj = lambda xc: jnp.sum((xc - _center(ints, theta)) ** 2)
        return run_sqp(neg_obj, x0, LB, UB, CFG)

    per_combo = jax.vmap(solve_cell, in_axes=(0, 0, None))   # integer dim, lower
    per_theta = jax.vmap(per_combo, in_axes=(0, None, 0))    # theta cells, higher

    ints = jnp.asarray([0.0, 1.0, 2.0, 3.0])                 # (C,)
    theta = jnp.asarray([0.5, 1.0, 1.5])                     # (M,)
    M, C = theta.shape[0], ints.shape[0]
    rng = np.random.default_rng(2)
    x0 = jnp.asarray(rng.uniform(0.05, 0.95, size=(M, C, D)))

    states = jax.jit(per_theta)(x0, ints, theta)

    assert states.x.shape == (M, C, D)
    assert bool(states.converged.all())
    expected = jax.vmap(
        lambda t: jax.vmap(lambda k: _center(k, t))(ints)
    )(theta)
    np.testing.assert_allclose(states.x, expected, atol=1e-6)
    np.testing.assert_allclose(states.f_val, 0.0, atol=1e-10)

    x_best, f_best, _ = best_cell(states)
    np.testing.assert_allclose(f_best, 0.0, atol=1e-10)
    print("test_nested_vmap_closure PASSED")


def test_closure_matches_p_threading():
    """The closure path must agree with the old p-threading path on the
    same conditioned family: factory solves the flat (M*C) batch with
    p=(ints, theta); the driver solves the same cells via nested vmap."""

    def f_xp(x, p):
        return jnp.sum((x - _center(p[0], p[1])) ** 2) + 0.1 * jnp.sum(
            jnp.cos(3.0 * x)
        )

    problem = ParametricNLPProblem(
        objective=f_xp, bounds=[LB, UB], n_decision=D, n_params=2,
    )
    factory = ParametricSQPFactory(problem, CFG)

    ints = jnp.asarray([0.0, 1.0, 2.0])
    theta = jnp.asarray([0.4, 0.9])
    M, C = theta.shape[0], ints.shape[0]
    rng = np.random.default_rng(3)
    x0 = jnp.asarray(rng.uniform(0.05, 0.95, size=(M, C, D)))

    p_flat = jnp.stack(
        [jnp.repeat(ints[None, :], M, 0).reshape(-1),
         jnp.repeat(theta[:, None], C, 1).reshape(-1)],
        axis=1,
    )  # (M*C, 2) — theta-major to match x0.reshape(M*C, D)
    ref = factory.solve_batch(x0.reshape(-1, D), p_flat)

    def solve_cell(x0_, k, t):
        neg_obj = lambda xc: f_xp(xc, jnp.stack([k, t]))
        return run_sqp(neg_obj, x0_, LB, UB, CFG)

    per_combo = jax.vmap(solve_cell, in_axes=(0, 0, None))
    per_theta = jax.vmap(per_combo, in_axes=(0, None, 0))
    got = jax.jit(per_theta)(x0, ints, theta)

    np.testing.assert_array_equal(
        np.asarray(got.converged).reshape(-1), np.asarray(ref.success)
    )
    np.testing.assert_allclose(
        got.x.reshape(-1, D), ref.decision_variables, atol=1e-10
    )
    np.testing.assert_allclose(got.f_val.reshape(-1), ref.objective, atol=1e-10)
    print("test_closure_matches_p_threading PASSED")


if __name__ == "__main__":
    test_unconstrained_parity()
    test_constrained_parity()
    test_nested_vmap_closure()
    test_closure_matches_p_threading()
    print("ALL PASSED")
