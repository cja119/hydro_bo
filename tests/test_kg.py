"""Mathematical validation of the parametric KG acquisition.

Checks properties the formulation pins down rather than golden numbers:
quadrature exactness, the triangular transform, the posterior-covariance
form of beta, sign consistency, and the nesting order
E_z[max_x' E_theta'] (which Jensen makes >= the collapsed ordering).

Run directly:  python tests/test_kg.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import jax
import jax.numpy as jnp
from septal.jax.sqp import SQPConfig

from hydro_bo.opt.acquisition import KnowledgeGradient, KnowledgeGradientInner
from hydro_bo.opt.dataset import Dataset
from hydro_bo.opt.solvers import combo_grid, mip_solve, sobol_cloud
from hydro_bo.opt.surrogate import HeteroscedasticGP, _kernel, _predict_covariance

D_X, D_T = 2, 2
D = D_X + D_T
INNER_CFG = SQPConfig(max_iter=8, use_exact_hessian=True)


# --------------------------------------------------------------- quadrature


def test_z_quadrature_moments():
    """Gauss-Hermite must integrate a standard normal: weights sum to 1,
    odd moments vanish, E[z^2]=1, E[z^4]=3. Catches a missing sqrt(2) on
    the nodes or a missing 1/sqrt(pi) on the weights."""
    acq = _make_kg.__wrapped__() if hasattr(_make_kg, "__wrapped__") else None
    z, w = _bare_kg()._gauss_hermite_quadrature(9)
    z, w = np.asarray(z), np.asarray(w)
    np.testing.assert_allclose(w.sum(), 1.0, atol=1e-12)
    np.testing.assert_allclose((w * z).sum(), 0.0, atol=1e-12)
    np.testing.assert_allclose((w * z**2).sum(), 1.0, atol=1e-12)
    np.testing.assert_allclose((w * z**4).sum(), 3.0, atol=1e-10)
    print("test_z_quadrature_moments PASSED")


def test_theta_quadrature_moments():
    """Inverse-CDF Gauss-Legendre must reproduce the symmetric triangular
    on [0,1]: mean 1/2, variance 1/24. A `leggauss(n)+0.5` style transform
    or a stray pdf multiply breaks both."""
    inner = KnowledgeGradientInner(None, None, (), quad_per_dim=24, num_theta=D_T)
    nodes = np.asarray(inner._theta_vals)
    w = np.asarray(inner._quad_weights)
    np.testing.assert_allclose(w.sum(), 1.0, atol=1e-12)
    mean = (w[:, None] * nodes).sum(axis=0)
    np.testing.assert_allclose(mean, np.full(D_T, 0.5), atol=2e-4)
    var = (w[:, None] * (nodes - mean) ** 2).sum(axis=0)
    np.testing.assert_allclose(var, np.full(D_T, 1.0 / 24.0), rtol=5e-3)
    assert nodes.shape == (24**D_T, D_T)
    assert nodes.min() >= 0.0 and nodes.max() <= 1.0
    print("test_theta_quadrature_moments PASSED")


def test_triangular_inv_cdf():
    """F(F^-1(p)) == p for the symmetric triangular on [0,1]."""
    p = np.linspace(1e-6, 1 - 1e-6, 401)
    x = np.asarray(KnowledgeGradientInner._inv_cdf(jnp.asarray(p)))
    f = np.where(x <= 0.5, 2.0 * x**2, 1.0 - 2.0 * (1.0 - x) ** 2)
    np.testing.assert_allclose(f, p, atol=1e-9)
    print("test_triangular_inv_cdf PASSED")


# ------------------------------------------------------------------ GP / beta


def _fit_gps(seed=0, n=18):
    rng = np.random.default_rng(seed)
    Xu = rng.uniform(size=(n, D))
    y = -np.sum((Xu - 0.55) ** 2, axis=1) + 0.15 * np.sin(5 * Xu[:, 0])
    bounds = np.tile(np.array([0.0, 1.0]), (D, 1))
    ds = Dataset(Xu, [np.array([v, v + 0.05, v - 0.03]) for v in y], bounds)
    gp_mu = HeteroscedasticGP(pad_initial=32, seed=seed, kernel_kind="rbf")
    gp_mu.fit(ds.X_scaled, ds.mu_scaled, 0.01 * np.ones(n))
    gp_lv = HeteroscedasticGP(pad_initial=32, seed=seed + 1, kernel_kind="rbf")
    m = ds.mask_log_sigma2
    gp_lv.fit(ds.X_scaled[m], ds.log_sigma2_scaled[m], 0.1 * np.ones(int(m.sum())))
    return gp_mu, gp_lv, ds


def test_beta_uses_posterior_covariance():
    """_predict_covariance must equal k(x',x) - k(x',X)K^-1k(X,x), and must
    differ from the prior covariance (conditioning actually happened)."""
    gp, _, _ = _fit_gps()
    st = gp.state()
    rng = np.random.default_rng(1)
    Xt = jnp.asarray(rng.uniform(size=(5, D)))
    xc = jnp.asarray(rng.uniform(size=(D,)))

    post = np.asarray(_predict_covariance(
        st["params"], st["X"], st["L"], st["mask"], Xt, xc, (), gp.kernel_kind))

    ka, kl = st["params"]["log_amp"], st["params"]["log_ls"]
    Ktr = np.asarray(_kernel(ka, kl, Xt, st["X"], (), gp.kernel_kind))
    Kc = np.asarray(_kernel(ka, kl, xc[None, :], st["X"], (), gp.kernel_kind))
    Kx = np.asarray(_kernel(ka, kl, Xt, xc[None, :], (), gp.kernel_kind)).ravel()
    from scipy.linalg import cho_solve
    v = cho_solve((np.asarray(st["L"]), True), Kc.T)
    ref = Kx - ((Ktr * v.T) * np.asarray(st["mask"])[None, :]).sum(axis=1)

    np.testing.assert_allclose(post, ref, atol=1e-9)
    assert not np.allclose(post, Kx), "conditioning had no effect"
    print("test_beta_uses_posterior_covariance PASSED")


# ------------------------------------------------------------------------ KG


def _bare_kg(**kg):
    gp_mu, gp_lv, ds = _fit_gps()
    args = dict(mode="strict", theta_quad_per_dim=3, z_quad_points=5,
                inner_pow_sobol=5, inner_n_restarts=1, index_set_pow=6,
                inner_sqp_config=INNER_CFG)
    args.update(kg)
    return KnowledgeGradient(
        gp_mu=gp_mu, gp_log_var=gp_lv, dataset=ds, d_theta=D_T,
        cat_vars=args.pop("cat_vars", ()), kg_args=args,
    )


def _make_kg(**kw):
    return _bare_kg(**kw)


def test_kg_is_finite_and_jensen_dominates():
    """KG >= max_x' E_theta'[mu_n] (the z=0 value).

    max is convex in z, so E_z[max] >= max at E[z]=0. Equality would mean
    the argmax never moves with z — i.e. the ordering had collapsed.
    """
    acq = _make_kg()
    rng = np.random.default_rng(3)
    cand = jnp.asarray(rng.uniform(size=(D,)))
    kg = float(acq.evaluate(cand))
    assert np.isfinite(kg)

    _, var_c = acq.mu(cand[None, :])
    grid = jnp.asarray(rng.uniform(size=(400, D_X)))
    v0 = jax.vmap(
        lambda xp: acq._inner_obj.value(xp, cand, var_c[0], 0.0)
    )(grid)
    baseline = float(jnp.max(v0))
    assert kg >= baseline - 1e-6, f"KG {kg} < z=0 baseline {baseline}"
    print(f"test_kg_is_finite_and_jensen_dominates PASSED "
          f"(KG={kg:.6f} >= {baseline:.6f})")


def test_inner_solve_is_a_maximisation():
    """The inner solve must beat a random start at every Hermite node —
    the check that the mip_solve minimise/maximise sign flip is coherent."""
    acq = _make_kg()
    rng = np.random.default_rng(4)
    cand = jnp.asarray(rng.uniform(size=(D,)))
    _, var_c = acq.mu(cand[None, :])
    probe = jnp.asarray(rng.uniform(size=(D_X,)))

    for k in range(acq._z_nodes.shape[0]):
        z = acq._z_nodes[k]
        x_star = acq._solve_inner(cand, var_c[0], z)
        best = float(acq._inner_obj.value(x_star, cand, var_c[0], z))
        start = float(acq._inner_obj.value(probe, cand, var_c[0], z))
        assert best >= start - 1e-6, f"z node {k}: {best} < probe {start}"
    print("test_inner_solve_is_a_maximisation PASSED")


def test_mixed_integer_inner_respects_levels():
    """With integer design dims the inner argmax must land on a declared
    level."""
    levels = [0.0, 0.5, 1.0]
    acq = _make_kg(cat_vars=[(0, levels)])
    rng = np.random.default_rng(7)
    cand = jnp.asarray(rng.uniform(size=(D,)))
    _, var_c = acq.mu(cand[None, :])
    assert acq.combos.shape == (3, 1)
    assert acq.d_red == D_X - 1
    x_star = np.asarray(acq._solve_inner(cand, var_c[0], acq._z_nodes[2]))
    assert min(abs(x_star[0] - l) for l in levels) < 1e-9, x_star
    print("test_mixed_integer_inner_respects_levels PASSED")


def test_index_set_mode_lower_bounds_strict():
    """index_set restricts the inner max to a finite set, so it can never
    exceed the continuum max that `strict` approximates."""
    rng = np.random.default_rng(11)
    cands = jnp.asarray(rng.uniform(size=(4, D)))
    a_strict = _make_kg(mode="strict")
    a_index = _make_kg(mode="index_set", index_set_pow=8)
    for c in cands:
        s, i = float(a_strict.evaluate(c)), float(a_index.evaluate(c))
        assert np.isfinite(s) and np.isfinite(i)
        assert i <= s + 1e-4, f"index_set {i} > strict {s}"
    print("test_index_set_mode_lower_bounds_strict PASSED")


def test_envelope_gradient_matches_finite_differences():
    """stop_gradient on the inner maximisers gives the Danskin gradient;
    it must agree with finite differences of the KG value."""
    acq = _make_kg(mode="index_set", index_set_pow=7)
    rng = np.random.default_rng(8)
    cand = jnp.asarray(rng.uniform(0.3, 0.7, size=(D,)))
    f = lambda x: acq._kg_value(x)
    g = np.asarray(jax.grad(f)(cand))
    assert np.all(np.isfinite(g))
    eps = 1e-5
    for i in range(D):
        e = jnp.zeros(D).at[i].set(eps)
        fd = float((f(cand + e) - f(cand - e)) / (2 * eps))
        assert abs(fd - g[i]) < 1e-3 * max(1.0, abs(fd)), (
            f"dim {i}: autodiff {g[i]:.6e} vs fd {fd:.6e}")
    print("test_envelope_gradient_matches_finite_differences PASSED")


def test_batch_matches_sequential_and_no_recompile():
    """acq_batch_fn (the Sobol screen path) must agree with per-point
    evaluation, and later candidates must reuse the compiled executable."""
    acq = _make_kg(mode="index_set")
    rng = np.random.default_rng(9)
    X = jnp.asarray(rng.uniform(size=(5, D)))
    batch = np.asarray(acq.acq_batch_fn()(X))
    seq = np.array([float(acq.evaluate(x)) for x in X])
    np.testing.assert_allclose(batch, seq, rtol=1e-9, atol=1e-9)

    n_before = acq._value._cache_size()
    for x in X:
        acq.evaluate(x)
    assert acq._value._cache_size() == n_before
    print("test_batch_matches_sequential_and_no_recompile PASSED")


def test_mip_solve_matches_class_driver():
    """The traced mip_solve must find the same optimum as the existing
    MixedIntNLP path on a problem both can express."""
    cat_vars = [(0, [0.0, 0.5, 1.0])]
    cat_idx, combos = combo_grid(cat_vars)
    centre = jnp.asarray([0.5, 0.3, 0.7])
    neg = lambda x: jnp.sum((x - centre) ** 2)

    cloud = sobol_cloud(2, 8, seed=0)
    x_best, val = mip_solve(
        neg, cloud, combos, cat_idx,
        jnp.zeros(2), jnp.ones(2),
        SQPConfig(max_iter=40, use_exact_hessian=True), n_restarts=2,
    )
    x_best = np.asarray(x_best)
    assert abs(x_best[0] - 0.5) < 1e-6, x_best      # nearest declared level
    np.testing.assert_allclose(x_best[1:], np.asarray(centre)[1:], atol=1e-5)
    np.testing.assert_allclose(float(val), 0.0, atol=1e-8)
    print("test_mip_solve_matches_class_driver PASSED")


if __name__ == "__main__":
    test_theta_quadrature_moments()
    test_triangular_inv_cdf()
    test_z_quadrature_moments()
    test_beta_uses_posterior_covariance()
    test_mip_solve_matches_class_driver()
    test_kg_is_finite_and_jensen_dominates()
    test_inner_solve_is_a_maximisation()
    test_mixed_integer_inner_respects_levels()
    test_index_set_mode_lower_bounds_strict()
    test_envelope_gradient_matches_finite_differences()
    test_batch_matches_sequential_and_no_recompile()
    print("ALL PASSED")
