"""Migration parity: the direct-driver solver classes must reproduce the
factory-path results (golden values captured pre-migration) on a
synthetic EI / cEI problem with two integer dims.

Run directly (no pytest needed):  python tests/test_solver_migration.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from septal.jax.sqp import SQPConfig
from hydro_bo.opt.dataset import Dataset
from hydro_bo.opt.surrogate import HeteroscedasticGP, BinomialGP
from hydro_bo.opt.acquisition import (
    ExpectedImprovement,
    ConstrainedExpectedImprovement,
)
from hydro_bo.opt.solvers import MixedIntNLP, ConstrainedMixedIntNLP

D = 5
BOUNDS = np.array([[0.0, 10.0], [1.0, 4.0], [0.0, 1.0], [0.0, 5.0], [2.0, 8.0]])
CAT_VARS = [(1, [0.0, 1 / 3, 2 / 3, 1.0]), (4, [0.0, 0.5, 1.0])]
ROUND_INFO = ((1, 4), (4, 3))
CFG = SQPConfig(max_iter=60, use_exact_hessian=True,
                tol_stationarity=1e-6, tol_feasibility=1e-6)

GOLDEN_UNCON_X = [0.334254679768682, 0.6666666666666666, 0.7355133220384793,
                  0.4752900191323169, 0.5]
GOLDEN_UNCON_F = 0.21696644317296607
GOLDEN_CON_BINDING_X = [0.7298132579092115, 0.6666666666666666,
                        0.9234764606162459, 0.11347731298235486, 0.5]
GOLDEN_CON_BINDING_F = 2.8473169367377774e-35
GOLDEN_CON_INACTIVE_X = [0.33425467976868206, 0.6666666666666666,
                         0.7355133220384793, 0.47529001913231705, 0.5]
GOLDEN_CON_INACTIVE_F = 0.21696644317296546


def synth_problem(seed=0):
    rng = np.random.default_rng(seed)
    n = 24
    Xu = rng.uniform(0, 1, size=(n, D))
    X = BOUNDS[:, 0] + Xu * (BOUNDS[:, 1] - BOUNDS[:, 0])
    samples = []
    for i in range(n):
        base = (
            -np.sum((Xu[i] - 0.6) ** 2)
            + 0.3 * np.sin(6 * Xu[i, 0])
            + 0.2 * Xu[i, 1]
        )
        s = base + 0.15 * rng.standard_normal(6)
        if Xu[i, 3] > 0.8:
            s[rng.integers(0, 6, size=2)] = np.nan
        samples.append(s)
    return Dataset(X, samples, BOUNDS)


def fit_gps(ds, seed=0):
    Xu = ds.X_scaled
    m_lv = ds.mask_log_sigma2
    gp_lv = HeteroscedasticGP(pad_initial=32, seed=seed, kernel_kind="rbf")
    gp_lv.fit(Xu[m_lv], ds.log_sigma2_scaled[m_lv],
              0.1 * np.ones(m_lv.sum()), round_info=ROUND_INFO)
    m_mu = ds.mask_mu
    gp_mu = HeteroscedasticGP(pad_initial=32, seed=seed, kernel_kind="rbf")
    noise = np.where(np.isfinite(ds.sigma2[m_mu]), ds.sigma2[m_mu], 0.05)
    noise = noise / (ds._sigma_y ** 2) / np.maximum(ds.k[m_mu], 1)
    gp_mu.fit(Xu[m_mu], ds.mu_scaled[m_mu], noise, round_info=ROUND_INFO)
    gp_bin = BinomialGP(pad_initial=32, seed=seed, label_smoothing=0.5)
    gp_bin.fit(Xu, ds.k, ds.N, round_info=ROUND_INFO)
    return gp_mu, gp_lv, gp_bin


def _setup():
    ds = synth_problem()
    return ds, *fit_gps(ds)


def test_unconstrained_golden(setup):
    ds, gp_mu, gp_lv, _ = setup
    ei = ExpectedImprovement(gp_mu, gp_lv, 0.5, ds, round_info=ROUND_INFO)
    solver = MixedIntNLP(cat_vars=CAT_VARS, seed=7, pow_sobol=8,
                         n_restarts=3, sqp_config=CFG)
    x, f = solver.maximise(ei)
    np.testing.assert_allclose(x, GOLDEN_UNCON_X, atol=1e-8)
    np.testing.assert_allclose(f, GOLDEN_UNCON_F, rtol=1e-8)
    print("test_unconstrained_golden PASSED")


def _constrained(setup, p_targ, z_sc, golden_x, golden_f):
    ds, gp_mu, gp_lv, gp_bin = setup
    cei = ConstrainedExpectedImprovement(
        gp_mu, gp_lv, gp_bin, 0.5, p_targ=p_targ, z_sc=z_sc,
        dataset=ds, round_info=ROUND_INFO,
    )
    solver = ConstrainedMixedIntNLP(cat_vars=CAT_VARS, l1_penalty=50.0,
                                    seed=7, pow_sobol=8, n_restarts=3,
                                    sqp_config=CFG)
    x, f = solver.maximise(cei)
    np.testing.assert_allclose(x, golden_x, atol=1e-8)
    np.testing.assert_allclose(f, golden_f, rtol=1e-6, atol=1e-40)


def test_constrained_binding_golden(setup):
    _constrained(setup, 0.8, 1.0, GOLDEN_CON_BINDING_X, GOLDEN_CON_BINDING_F)
    print("test_constrained_binding_golden PASSED")


def test_constrained_inactive_golden(setup):
    _constrained(setup, 0.65, 1.0, GOLDEN_CON_INACTIVE_X, GOLDEN_CON_INACTIVE_F)
    print("test_constrained_inactive_golden PASSED")


try:
    import pytest

    @pytest.fixture(scope="module", name="setup")
    def _setup_fixture():
        return _setup()
except ImportError:
    pass


if __name__ == "__main__":
    s = _setup()
    test_unconstrained_golden(s)
    test_constrained_binding_golden(s)
    test_constrained_inactive_golden(s)
    print("ALL PASSED")
