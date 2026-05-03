"""
Bayesian Optimiser with heteroscedastic noise modelling and EI.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
from scipy.stats.qmc import Sobol

from hydro_bo.algs.logging_config import get_logger

logger = get_logger(__name__)


# Enabling parallelism for JAX-operations (call before importing jax)
def configure_jax_threads(n_threads: int) -> None:
    import sys
    if "jax" in sys.modules:
        logger.warning("configure_jax_threads_late",
                       message="jax already imported — XLA flags will not take effect",
                       n_threads=int(n_threads))
        return
    n = max(1, int(n_threads))
    flags = [
        "--xla_cpu_multi_thread_eigen=true",
        f"intra_op_parallelism_threads={n}",
        f"inter_op_parallelism_threads={n}",
    ]
    existing = os.environ.get("XLA_FLAGS", "")
    os.environ["XLA_FLAGS"] = (existing + " " + " ".join(flags)).strip()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(n)

# Jax imports are deferred
def _jax():
    import jax
    return jax

def _jnp():
    import jax.numpy as jnp
    return jnp

# The data handler is used to manage some of the more challenging
# properties of the data:
# - Raggedness (variable number of samples per point)
# - Standardisation (z-scoring of targets for GP fitting)
# - Unit-cube scaling of X for the ARD RBF kernel.
@dataclass
class Dataset:
    _X: np.ndarray
    _samples: list
    bounds: np.ndarray

    def __post_init__(self):
        # Processing the bounds
        b = np.asarray(self.bounds, dtype=float)
        self._lo = b[:, 0]
        self._span = b[:, 1] - b[:, 0]

        # Processing samples and the number of evaluations
        samples = [np.asarray(s, dtype=float).ravel() for s in self._samples]
        self._samples = samples
        self.n = np.array([len(s) for s in samples], dtype=int)

        with np.errstate(invalid="ignore"):
            # When we have enough samples we compute the sample mean and variance
            # which will be used in GP training
            self.mu = np.array([
                float(np.mean(s)) if len(s) >= 1 else np.nan for s in samples
            ])
            self.sigma2 = np.array([
                float(np.var(s, ddof=1)) if len(s) >= 2 else np.nan
                for s in samples
            ])

        # Computing a numerically stable log-variance
        floor = 1e-12
        log_v = np.full_like(self.sigma2, np.nan)
        valid_v = np.isfinite(self.sigma2) & (self.sigma2 > 0)
        log_v[valid_v] = np.log(np.maximum(self.sigma2[valid_v], floor))
        self.log_sigma2 = log_v

        # Computing scaling parameters for mean GP target.
        finite_mu = self.mu[np.isfinite(self.mu)]
        self._mu_y = float(np.mean(finite_mu)) if finite_mu.size else 0.0
        sy = float(np.std(finite_mu)) if finite_mu.size else 1.0
        self._sigma_y = sy if sy > 0 else 1.0
        
        # Computing scaling parameters for log-variance GP target.
        finite_lv = self.log_sigma2[np.isfinite(self.log_sigma2)]
        self._mu_lv = float(np.mean(finite_lv)) if finite_lv.size else 0.0
        slv = float(np.std(finite_lv)) if finite_lv.size else 1.0
        self._sigma_lv = slv if slv > 0 else 1.0

    @property
    def X(self):
        """Original X in bounds (shape n x d)."""
        return self._X

    @property
    def X_scaled(self):
        """Unit-cube scaled X."""
        return (self._X - self._lo) / self._span

    def to_unit(self, X: np.ndarray) -> np.ndarray:
        """Scale X in bounds to the unit cube."""
        return (X - self._lo) / self._span

    def to_original(self, X_unit: np.ndarray) -> np.ndarray:
        """Scale X from the unit cube back to original bounds."""
        return self._lo + X_unit * self._span

    @property
    def mu_scaled(self):
        """Scaled mean targets for the mean GP."""
        return (self.mu - self._mu_y) / self._sigma_y

    @property
    def log_sigma2_scaled(self):
        """Scaled log-variance targets for the log-variance GP."""
        return (self.log_sigma2 - self._mu_lv) / self._sigma_lv

    @property
    def mask_mu(self):
        """Rows usable by the mean GP (>= 1 valid sample)."""
        return np.isfinite(self.mu)

    @property
    def mask_log_sigma2(self):
        """Rows usable by the log-variance GP (>= 2 valid samples, s^2 > 0)."""
        return np.isfinite(self.log_sigma2)

    @property
    def noise_log_sigma2(self):
        """Approx Var[log s^2] ~= 2/(n-1) (chi-square asymptotic)."""
        n = self.n.astype(float)
        return np.where(n > 1, 2.0 / np.maximum(n - 1, 1.0), np.nan)


def _project_integer_dims(X, round_info):
    """Snap the integer dims of X to their nearest unit-cube grid position.
    round_info: sorted tuple of (dim_idx, n_levels) — both static Python ints.
    Empty round_info is a no-op (returns X unchanged)."""
    if not round_info:
        return X
    jnp = _jnp()
    for idx, n in round_info:
        if n <= 1:
            continue
        denom = n - 1
        col = X[..., idx]
        rounded = jnp.round(col * denom) / denom
        X = X.at[..., idx].set(rounded)
    return X


class BaseGP(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, noise: np.ndarray,
            round_info: tuple = ()) -> None:
        """Fit the GP to data (X, y) with per-point noise variances."""
        ...

    @abstractmethod
    def predict(self, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict (mean, var) at X_test using the round_info from fit time."""
        ...

class HeteroscedasticGP(BaseGP):
    """Exact GP regression with constant mean, ARD RBF kernel, and fixed
    per-point observation noise. Hyperparameters (kernel amplitude,
    lengthscales, mean) are fit by L-BFGS on the negative marginal
    log-likelihood with JAX autodiff."""

    def __init__(self, jitter: float = 1e-6, max_iters: int = 75):
        self.jitter = jitter
        self.max_iters = max_iters
        self.params = None
        self._X = None
        self._L = None
        self._alpha = None
        self._mean = None
        self.round_info: tuple = ()

    @staticmethod
    def _kernel(log_amp, log_ls, X1, X2, round_info):
        """ARD RBF kernel with optional projection of integer dims onto a
        discrete unit-cube grid. round_info is a static Python tuple — empty
        skips projection."""
        if round_info:
            X1 = _project_integer_dims(X1, round_info)
            X2 = _project_integer_dims(X2, round_info)
        jnp = _jnp()
        amp2 = jnp.exp(2.0 * log_amp)
        ls = jnp.exp(log_ls)
        diff = (X1[:, None, :] - X2[None, :, :]) / ls
        d2 = jnp.sum(diff * diff, axis=-1)
        return amp2 * jnp.exp(-0.5 * d2)

    @staticmethod
    def _neg_mll_static(params, X, y, noise, jitter, round_info):
        jnp = _jnp()
        jax = _jax()
        n = X.shape[0]
        K = HeteroscedasticGP._kernel(
            params["log_amp"], params["log_ls"], X, X, round_info
        )
        K = K + jnp.diag(noise) + jitter * jnp.eye(n)
        y_c = y - params["mean"]
        L = jnp.linalg.cholesky(K)
        alpha = jax.scipy.linalg.cho_solve((L, True), y_c)
        log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
        return 0.5 * (jnp.dot(y_c, alpha) + log_det + n * jnp.log(2.0 * jnp.pi))

    @staticmethod
    def _factorise_static(params, X, y, noise, jitter, round_info):
        jnp = _jnp()
        jax = _jax()
        n = X.shape[0]
        K = HeteroscedasticGP._kernel(
            params["log_amp"], params["log_ls"], X, X, round_info
        )
        K = K + jnp.diag(noise) + jitter * jnp.eye(n)
        L = jnp.linalg.cholesky(K)
        alpha = jax.scipy.linalg.cho_solve((L, True), y - params["mean"])
        return L, alpha

    @staticmethod
    def _predict_static(params, X_train, L, alpha, mean, X_test, round_info):
        jnp = _jnp()
        jax = _jax()
        K_s = HeteroscedasticGP._kernel(
            params["log_amp"], params["log_ls"], X_test, X_train, round_info
        )
        amp2 = jnp.exp(2.0 * params["log_amp"])
        mu = mean + K_s @ alpha
        v = jax.scipy.linalg.cho_solve((L, True), K_s.T)
        var = amp2 - jnp.sum(K_s * v.T, axis=1)
        var = jnp.clip(var, 1e-12, None)
        return mu, var

    def fit(self, X, y, noise, round_info: tuple = ()):
        from jaxopt import LBFGS
        jnp = _jnp()
        X = jnp.asarray(X, dtype=jnp.float64)
        y = jnp.asarray(y, dtype=jnp.float64).reshape(-1)
        noise = jnp.asarray(noise, dtype=jnp.float64).reshape(-1)
        n, d = int(X.shape[0]), int(X.shape[1])
        round_info = tuple(round_info)
        self.round_info = round_info

        logger.debug("gp_fit_start", n_datapoints=n, n_dims=d,
                     warm_start=self.params is not None)

        if self.params is not None and self.params["log_ls"].shape[0] == d:
            init = {
                "log_amp": self.params["log_amp"],
                "log_ls": self.params["log_ls"],
                "mean": self.params["mean"],
            }
        else:
            init = {
                "log_amp": jnp.array(0.0, dtype=jnp.float64),
                "log_ls": jnp.zeros(d, dtype=jnp.float64),
                "mean": jnp.array(float(np.mean(np.asarray(y))), dtype=jnp.float64),
            }

        jitter = jnp.asarray(self.jitter, dtype=jnp.float64)
        nmll = _get_jitted_nmll(round_info)
        solver = LBFGS(fun=nmll, maxiter=self.max_iters)
        result = solver.run(init, X, y, noise, jitter)
        self._release_buffers()
        self.params = {k: jnp.asarray(v) for k, v in result.params.items()}

        L, alpha = _get_jitted_factorise(round_info)(
            self.params, X, y, noise, jitter
        )
        self._X = X
        self._L = L
        self._alpha = alpha
        self._mean = self.params["mean"]

        logger.debug("gp_fit_complete",
                     final_nmll=float(result.state.value),
                     n_lbfgs_iters=int(result.state.iter_num))

    def _release_buffers(self):
        for attr in ("_X", "_L", "_alpha", "_mean"):
            buf = getattr(self, attr, None)
            if buf is not None and hasattr(buf, "delete"):
                try:
                    buf.delete()
                except Exception:
                    pass
            setattr(self, attr, None)

    def predict(self, X_test):
        """Returns (mean, var) at X_test in standardised target space.
        Uses self.round_info captured at fit time so predictions are
        kernel-consistent with the fit."""
        jnp = _jnp()
        X_test = jnp.asarray(X_test, dtype=jnp.float64)
        if X_test.ndim == 1:
            X_test = X_test[None, :]
        return _get_jitted_predict(self.round_info)(
            self.params, self._X, self._L, self._alpha, self._mean, X_test
        )


# Module-level jit handles, keyed by round_info tuple. Each entry caches the
# Python jit object specialised on a particular round_info pattern. round_info
# is closure-captured (static) so JAX bakes the projection ops into the trace;
# mask_values for branch-no-bound are kept as runtime args so one compile
# per (mask, round) pair serves every combo within an iteration.
_JIT_NMLL_BY_ROUND: dict = {}
_JIT_FACTORISE_BY_ROUND: dict = {}
_JIT_PREDICT_BY_ROUND: dict = {}
_JIT_ACQ_G_STATS_BY_ROUND: dict = {}
_JIT_ACQ_EVAL_BY_ROUND: dict = {}
_JIT_ACQ_BATCH_BY_ROUND: dict = {}
_JIT_NEG_ACQ_BY_ROUND: dict = {}

# Branch-no-bound: keyed by (mask_indices, round_info).
_JIT_MASKED_NEG_ACQ_BY_MASK: dict = {}
_JIT_MASKED_ACQ_BATCH_BY_MASK: dict = {}


def _get_jitted_nmll(round_info):
    cache = _JIT_NMLL_BY_ROUND
    if round_info not in cache:
        jax = _jax()
        ri = round_info

        @jax.jit
        def nmll(params, X, y, noise, jitter):
            return HeteroscedasticGP._neg_mll_static(
                params, X, y, noise, jitter, ri
            )

        cache[round_info] = nmll
    return cache[round_info]


def _get_jitted_factorise(round_info):
    cache = _JIT_FACTORISE_BY_ROUND
    if round_info not in cache:
        jax = _jax()
        ri = round_info

        @jax.jit
        def factorise(params, X, y, noise, jitter):
            return HeteroscedasticGP._factorise_static(
                params, X, y, noise, jitter, ri
            )

        cache[round_info] = factorise
    return cache[round_info]


def _get_jitted_predict(round_info):
    cache = _JIT_PREDICT_BY_ROUND
    if round_info not in cache:
        jax = _jax()
        ri = round_info

        @jax.jit
        def predict(params, X_train, L, alpha, mean, X_test):
            return HeteroscedasticGP._predict_static(
                params, X_train, L, alpha, mean, X_test, ri
            )

        cache[round_info] = predict
    return cache[round_info]


def _acq_g_stats_pure(x, gp_mu_state, gp_log_var_state, scaling, round_info):
    """Pure version of NoisyEI._g_stats — all closure state passed explicitly
    so the per-round_info jit can serve every BO iteration."""
    jnp = _jnp()

    # Unpacking scaling parameters
    mu_y, sigma_y, mu_lv, sigma_lv, lam = scaling

    # Extracting the mean and log-variance GP states at x.
    mu_s, var_mu_s = HeteroscedasticGP._predict_static(
        gp_mu_state["params"], gp_mu_state["X"], gp_mu_state["L"],
        gp_mu_state["alpha"], gp_mu_state["mean"], x, round_info,
    )
    lv_s, var_lv_s = HeteroscedasticGP._predict_static(
        gp_log_var_state["params"], gp_log_var_state["X"], gp_log_var_state["L"],
        gp_log_var_state["alpha"], gp_log_var_state["mean"], x, round_info,
    )

    # Upscaled mean and variance for the mean GP.
    m_mu = mu_s * sigma_y + mu_y
    v_mu = var_mu_s * (sigma_y ** 2)

    # Upscaled mean and variance for the log-variance GP.
    m_lv = lv_s * sigma_lv + mu_lv
    v_lv = var_lv_s * (sigma_lv ** 2)

    # Exponentiating the log random variables.
    e_sd = jnp.exp(0.5 * m_lv + 0.125 * v_lv)
    var_sd = (jnp.exp(0.25 * v_lv) - 1.0) * jnp.exp(m_lv + 0.25 * v_lv)

    # Moment matching the distributions with the mean and standard deviations.
    e_g = m_mu - lam * e_sd
    var_g = v_mu + (lam ** 2) * var_sd

    return e_g, var_g


def _acq_eval_pure(x, gp_mu_state, gp_log_var_state, scaling, g_best, jitter,
                   round_info):
    """EI value at a single point. x has shape (d,)."""
    jnp = _jnp()
    from jax.scipy.stats.norm import cdf, pdf

    # Adding a dim for the acquisition evalution
    x2 = x[None, :]

    # Computing the acquisition function statistics
    e_g, var_g = _acq_g_stats_pure(
        x2, gp_mu_state, gp_log_var_state, scaling, round_info
    )
    sigma_g = jnp.sqrt(jnp.clip(var_g, jitter, None))

    # Computing the expected improvement
    delta = e_g - g_best
    z = delta / sigma_g
    ei = delta * cdf(z) + sigma_g * pdf(z)

    # Returning the acquisition value as a scalar
    return ei.squeeze()


def _neg_acq_pure(x, gp_mu_state, gp_log_var_state, scaling, g_best, jitter,
                  round_info):
    return -_acq_eval_pure(
        x, gp_mu_state, gp_log_var_state, scaling, g_best, jitter, round_info
    )


def _get_jitted_acq_g_stats(round_info):
    cache = _JIT_ACQ_G_STATS_BY_ROUND
    if round_info not in cache:
        jax = _jax()
        ri = round_info

        @jax.jit
        def acq_g_stats(x, gp_mu_state, gp_log_var_state, scaling):
            return _acq_g_stats_pure(
                x, gp_mu_state, gp_log_var_state, scaling, ri
            )

        cache[round_info] = acq_g_stats
    return cache[round_info]


def _get_jitted_acq_eval(round_info):
    cache = _JIT_ACQ_EVAL_BY_ROUND
    if round_info not in cache:
        jax = _jax()
        ri = round_info

        @jax.jit
        def acq_eval(x, gp_mu_state, gp_log_var_state, scaling, g_best, jitter):
            return _acq_eval_pure(
                x, gp_mu_state, gp_log_var_state, scaling, g_best, jitter, ri
            )

        cache[round_info] = acq_eval
    return cache[round_info]


def _get_jitted_acq_batch(round_info):
    cache = _JIT_ACQ_BATCH_BY_ROUND
    if round_info not in cache:
        jax = _jax()
        ri = round_info

        def acq_eval_with_round(x, gp_mu_state, gp_log_var_state, scaling,
                                g_best, jitter):
            return _acq_eval_pure(
                x, gp_mu_state, gp_log_var_state, scaling, g_best, jitter, ri
            )

        cache[round_info] = jax.jit(jax.vmap(
            acq_eval_with_round, in_axes=(0, None, None, None, None, None)
        ))
    return cache[round_info]


def _get_jitted_neg_acq(round_info):
    cache = _JIT_NEG_ACQ_BY_ROUND
    if round_info not in cache:
        jax = _jax()
        ri = round_info

        @jax.jit
        def neg_acq(x, gp_mu_state, gp_log_var_state, scaling, g_best, jitter):
            return _neg_acq_pure(
                x, gp_mu_state, gp_log_var_state, scaling, g_best, jitter, ri
            )

        cache[round_info] = neg_acq
    return cache[round_info]


def _get_jitted_masked_neg_acq(mask_indices: tuple, round_info: tuple):
    """Jitted neg-acq that inserts mask_values at mask_indices in x_reduced
    and applies kernel rounding via round_info. Cached by (mask_indices,
    round_info) — one compile per pair, served across every combo by passing
    mask_values as a JAX array argument."""
    cache = _JIT_MASKED_NEG_ACQ_BY_MASK
    key = (mask_indices, round_info)
    if key not in cache:
        jax = _jax()
        idxs = mask_indices
        ri = round_info

        def masked_neg_acq(x_reduced, mask_values, gp_mu_state, gp_log_var_state,
                           scaling, g_best, jitter):
            jnp = _jnp()
            x = x_reduced
            for i, idx in enumerate(idxs):
                left = x[..., :idx]
                right = x[..., idx:]
                fill = jnp.broadcast_to(
                    mask_values[i].astype(x.dtype), x.shape[:-1] + (1,)
                )
                x = jnp.concatenate([left, fill, right], axis=-1)
            return _neg_acq_pure(
                x, gp_mu_state, gp_log_var_state, scaling, g_best, jitter, ri
            )

        cache[key] = jax.jit(masked_neg_acq)
    return cache[key]


def _get_jitted_masked_acq_batch(mask_indices: tuple, round_info: tuple):
    """Jitted vmapped POSITIVE acq over a batch of x_reduced points; same
    mask_values broadcasting + round_info story as _get_jitted_masked_neg_acq."""
    cache = _JIT_MASKED_ACQ_BATCH_BY_MASK
    key = (mask_indices, round_info)
    if key not in cache:
        jax = _jax()
        idxs = mask_indices
        ri = round_info

        def expand_one(x_red_one, mask_values):
            jnp = _jnp()
            x = x_red_one
            for i, idx in enumerate(idxs):
                left = x[..., :idx]
                right = x[..., idx:]
                fill = jnp.broadcast_to(
                    mask_values[i].astype(x.dtype), x.shape[:-1] + (1,)
                )
                x = jnp.concatenate([left, fill, right], axis=-1)
            return x

        def acq_eval_with_round(x, gp_mu_state, gp_log_var_state, scaling,
                                g_best, jitter):
            return _acq_eval_pure(
                x, gp_mu_state, gp_log_var_state, scaling, g_best, jitter, ri
            )

        def masked_acq_batch(x_batch, mask_values, gp_mu_state, gp_log_var_state,
                             scaling, g_best, jitter):
            x_full_batch = jax.vmap(expand_one, in_axes=(0, None))(
                x_batch, mask_values
            )
            return jax.vmap(
                acq_eval_with_round, in_axes=(0, None, None, None, None, None)
            )(x_full_batch, gp_mu_state, gp_log_var_state, scaling, g_best, jitter)

        cache[key] = jax.jit(masked_acq_batch)
    return cache[key]


def _clear_jax_caches() -> None:
    """Drop XLA-compiled artifacts from the per-iteration GP / acquisition
    fits. Each BO iteration grows X by one row, which compiles a NEW XLA
    program under a unique shape key — without this, those programs pile
    up in the jit cache and consume hundreds of MB/iteration in the driver."""
    jax = _jax()
    try:
        jax.clear_caches()
    except AttributeError:
        try:
            from jax._src.compilation_cache import compilation_cache as _cc
            _cc.reset_cache()
        except Exception:
            pass


class AcquisitionFunction(ABC):
    @abstractmethod
    def evaluate(self, x):
        ...


class ExpectedImprovement(AcquisitionFunction):
    """
    Approximate EI for a heteroscedastic GP objective. Our objective is not 
    purely normal as g(x) = mu(x) - lam * sigma(x) with sigma(x) ~ LogNormal.

    Due to the scale difference of mu and sigma, we assume that g(x) is still 
    normally distributed, and match first and second order moments. (Assuming 
    independenceof mu and sigma).

    The acquisition is then EI with respect to the incumbent g_best = max_i E[g(x_i),
    but done over the surrogate function on the existing training dataset of x. Then 
    the standard expected improvement formula is applied.
    """

    def __init__(self, gp_mu: HeteroscedasticGP, gp_log_var: HeteroscedasticGP,
                 lam: float, dataset: Dataset, jitter: float = 1e-9,
                 round_info: tuple = ()):
        # round_info must match the round_info both GPs were fit with.
        # _fit_gps in the BO uses the same round_info for both fits, so this
        # is consistent by construction.
        jnp = _jnp()
        self.round_info = tuple(round_info)
        self.scaling = (
            jnp.asarray(dataset._mu_y, dtype=jnp.float64),
            jnp.asarray(dataset._sigma_y, dtype=jnp.float64),
            jnp.asarray(dataset._mu_lv, dtype=jnp.float64),
            jnp.asarray(dataset._sigma_lv, dtype=jnp.float64),
            jnp.asarray(lam, dtype=jnp.float64),
        )
        self.jitter = jnp.asarray(jitter, dtype=jnp.float64)
        self.gp_mu_state = {
            "params": gp_mu.params, "X": gp_mu._X, "L": gp_mu._L,
            "alpha": gp_mu._alpha, "mean": gp_mu._mean,
        }
        self.gp_log_var_state = {
            "params": gp_log_var.params, "X": gp_log_var._X, "L": gp_log_var._L,
            "alpha": gp_log_var._alpha, "mean": gp_log_var._mean,
        }
        self._g_best = self._incumbent(dataset)

    def _incumbent(self, dataset: Dataset):
        jnp = _jnp()
        m = dataset.mask_mu
        if int(m.sum()) == 0:
            return jnp.asarray(0.0, dtype=jnp.float64)
        X_unit = jnp.asarray(dataset.X_scaled[m], dtype=jnp.float64)
        e_g, _ = _get_jitted_acq_g_stats(self.round_info)(
            X_unit, self.gp_mu_state, self.gp_log_var_state, self.scaling
        )
        return jnp.max(e_g)

    def evaluate(self, x):
        return _get_jitted_acq_eval(self.round_info)(
            x, self.gp_mu_state, self.gp_log_var_state,
            self.scaling, self._g_best, self.jitter,
        )


def sobol_sample(bounds: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Draw n quasi-random points from bounds using Sobol sequence."""
    d = bounds.shape[0]
    sampler = Sobol(d=d, scramble=True, seed=seed)
    unit = sampler.random(n)
    lo, hi = bounds[:, 0], bounds[:, 1]
    return lo + unit * (hi - lo)


class BayesianOptimizer:
    """BO with stochastic objective f(x) -> sample array.

    State:
      _X        : list[np.ndarray] inputs in original bounds.
      _samples  : list[np.ndarray] per-input sample arrays (variable length).
      dataset   : last built Dataset (None until first fit).
    """

    def __init__(
        self,
        f: Callable,
        bounds: Sequence,
        n_initial_points: int,
        iter_limit: int,
        lam: float = 0.5,
        n_restarts: int = 5,
        pow_sobol: int = 14,  # 2**14 = 16384 acquisition Sobol candidates
        seed: int = 0,
        cat_vars: Sequence[Tuple[int, Sequence[float]]] = (),
    ):
        # cat_vars: (full_dim_index, list_of_unit_cube_positions). Values must
        # be in unit-cube coords — the branch optimiser inserts them straight
        # into the unit-cube x passed to the acquisition.
        self.f = f
        self.bounds = np.asarray(bounds, dtype=float)
        self.n_initial_points = n_initial_points
        self.iter_limit = iter_limit
        self.lam = lam
        self.n_restarts = n_restarts
        self.pow_sobol = pow_sobol
        self.seed = seed
        self.cat_vars = [(int(i), [float(v) for v in vals]) for i, vals in cat_vars]

        self.gp_mu = HeteroscedasticGP()
        self.gp_log_var = HeteroscedasticGP()
        self._X: list[np.ndarray] = []
        self._samples: list[np.ndarray] = []
        self.dataset: Dataset | None = None

    def _snap_to_grid(self, x: np.ndarray) -> np.ndarray:
        """Snap x's integer dims to their nearest unit-cube grid position."""
        if not self.cat_vars:
            return np.asarray(x, dtype=float)
        x = np.asarray(x, dtype=float).copy()
        for dim_idx, unit_positions in self.cat_vars:
            lo, hi = self.bounds[dim_idx]
            span = hi - lo
            if span <= 0:
                continue
            u = (x[dim_idx] - lo) / span
            positions = np.asarray(unit_positions, dtype=float)
            nearest_u = float(positions[int(np.argmin(np.abs(positions - u)))])
            x[dim_idx] = lo + nearest_u * span
        return x

    def observe(self, x: np.ndarray, samples) -> None:
        """Record an observation: x in original bounds, samples = 1D iterable.
        Integer dims of x are snapped to their grid before storage."""
        x = self._snap_to_grid(x)
        s = np.asarray(samples, dtype=float).ravel()
        s = s[np.isfinite(s)]
        self._X.append(x)
        self._samples.append(s)

    def _evaluate_and_store(self, x: np.ndarray) -> np.ndarray:
        x = self._snap_to_grid(x)
        s = np.asarray(self.f(x), dtype=float).ravel()
        s = s[np.isfinite(s)]
        self._X.append(x)
        self._samples.append(s)
        return s

    def run(self) -> tuple[np.ndarray, float]:
        n_preloaded = len(self._X)
        n_to_sample = max(0, self.n_initial_points - n_preloaded)
        if n_to_sample > 0:
            logger.info("bo_sobol_phase_start",
                        n_to_sample=n_to_sample, n_preloaded=n_preloaded)
            X_init = sobol_sample(self.bounds, n_to_sample, seed=self.seed)
            for i, x in enumerate(X_init):
                s = self._evaluate_and_store(x)
                logger.info("bo_sobol_observation",
                            point=i + 1, n_total=n_to_sample,
                            n_samples=int(len(s)),
                            mean=float(np.mean(s)) if len(s) else float("nan"))
        else:
            logger.info("bo_sobol_phase_skipped", n_preloaded=n_preloaded)

        logger.info("bo_phase_start", iter_limit=self.iter_limit, lam=self.lam)
        try:
            import psutil
            _proc = psutil.Process()
        except Exception:
            _proc = None
        for i in range(self.iter_limit):
            self._fit_gps()
            x_next = self._suggest()
            s = self._evaluate_and_store(x_next)
            best_x, best_score = self._best_observed()
            rss_mb = float(_proc.memory_info().rss) / (1024 ** 2) if _proc is not None else float("nan")
            logger.info(
                "bo_iteration",
                iteration=i + 1, iter_limit=self.iter_limit,
                n_samples=int(len(s)),
                mean=float(np.mean(s)) if len(s) else float("nan"),
                var=float(np.var(s, ddof=1)) if len(s) >= 2 else float("nan"),
                best_score=float(best_score),
                driver_rss_mb=rss_mb,
            )

        best_x, best_score = self._best_observed()
        logger.info("bo_complete", best_score=float(best_score),
                    n_total_evals=len(self._samples))
        return best_x, best_score

    def suggest(self) -> np.ndarray:
        self._fit_gps()
        return self._suggest()

    def _best_observed(self) -> tuple[np.ndarray, float]:
        """Best (x, mu - lam * s) over training points, where s is the
        sample stdev."""
        ds = Dataset(np.stack(self._X), self._samples, self.bounds)
        sd = np.where(np.isfinite(ds.sigma2), np.sqrt(np.maximum(ds.sigma2, 0.0)), 0.0)
        scores = ds.mu - self.lam * sd
        valid = np.isfinite(scores)
        if not valid.any():
            return self._X[0], float("nan")
        idx_in_valid = int(np.argmax(scores[valid]))
        idx = int(np.flatnonzero(valid)[idx_in_valid])
        return self._X[idx], float(scores[idx])

    def _build_round_info(self) -> tuple:
        """Derive the kernel round_info from self.cat_vars. Uses the integer-
        level count per categorical dim. Empty cat_vars → empty round_info,
        i.e. kernel stays continuous."""
        if not self.cat_vars:
            return ()
        info = sorted(
            ((int(idx), int(len(positions))) for idx, positions in self.cat_vars),
            key=lambda p: p[0],
        )
        return tuple(info)

    def _fit_gps(self) -> None:
        jnp = _jnp()
        X = np.stack(self._X)
        self.dataset = Dataset(X, self._samples, self.bounds)
        ds = self.dataset
        X_unit = ds.X_scaled
        round_info = self._build_round_info()

        # 1) Log-variance GP on rows with n >= 2.
        m_lv = ds.mask_log_sigma2
        log_var_fit = False
        if int(m_lv.sum()) >= 2:
            X_lv = X_unit[m_lv]
            y_lv = ds.log_sigma2_scaled[m_lv]
            noise_lv = ds.noise_log_sigma2[m_lv] / (ds._sigma_lv ** 2)
            self.gp_log_var.fit(X_lv, y_lv, noise_lv, round_info=round_info)
            log_var_fit = True
            logger.info("bo_log_var_gp_fit", n_points=int(m_lv.sum()))
        else:
            logger.warning("bo_log_var_gp_skipped",
                           n_valid=int(m_lv.sum()),
                           reason="need >= 2 points with n_samples >= 2")

        # 2) Mean GP on rows with n >= 1, with noise = predicted-pop-var / n
        # (fall back to empirical s^2 if log-var GP couldn't be fit yet).
        m_mu = ds.mask_mu
        if int(m_mu.sum()) < 2:
            raise RuntimeError(
                f"Mean GP needs >= 2 valid mean observations, got {int(m_mu.sum())}"
            )
        X_mu = X_unit[m_mu]
        y_mu = ds.mu_scaled[m_mu]
        n_mu = ds.n[m_mu].astype(float)

        if log_var_fit:
            log_v_pred_s, log_v_pred_v = self.gp_log_var.predict(
                jnp.asarray(X_mu, dtype=jnp.float64)
            )
            log_v_pred = np.asarray(log_v_pred_s) * ds._sigma_lv + ds._mu_lv
            log_v_pred_var = np.asarray(log_v_pred_v) * ds._sigma_lv ** 2
            pop_var = np.exp(log_v_pred + 1/2 * log_v_pred_var)
        else:
            sv = ds.sigma2[m_mu]
            mean_finite = float(np.nanmean(sv)) if np.isfinite(sv).any() else 1.0
            pop_var = np.where(np.isfinite(sv), sv, mean_finite)

        noise_mu = pop_var / np.maximum(n_mu, 1.0)
        noise_mu_scaled = noise_mu / (ds._sigma_y ** 2)
        self.gp_mu.fit(X_mu, y_mu, noise_mu_scaled, round_info=round_info)
        logger.info("bo_mean_gp_fit", n_points=int(m_mu.sum()),
                    used_log_var_gp=log_var_fit)

    def _suggest(self) -> np.ndarray:
        if self.dataset is None:
            self._fit_gps()
        logger.debug("bo_acquisition_optimise",
                     lam=self.lam, n_restarts=self.n_restarts)
        acq = ExpectedImprovement(
            self.gp_mu, self.gp_log_var, self.lam, self.dataset,
            round_info=self.gp_mu.round_info,
        )
        logger.debug("bo_noisy_ei_incumbent", g_best=float(acq._g_best))

        seed = self.seed + len(self._X)
        best_x_unit, best_val = _branch_no_bound_optimise_acquisition(
            acq, self.cat_vars, seed, self.pow_sobol, self.n_restarts,
        )

        logger.debug("bo_acquisition_optimise_complete", best_score=float(best_val))
        x_orig = self.dataset.to_original(np.asarray(best_x_unit))

        del acq
        _clear_jax_caches()
        import gc
        gc.collect()

        return x_orig
    

def _lbfgsb_maximise_acquisition(acq, x0, neg_acq=None, dim=None, extra_args=()):
    """Maximise acq over the unit cube via LBFGS-B. extra_args is forwarded as
    additional positional args to neg_acq before the GP/scaling state — used by
    the masked variants to thread mask_values through. When neg_acq is None,
    falls back to the round_info-aware neg-acq for acq.round_info."""
    from jaxopt import LBFGSB
    jnp = _jnp()
    d = dim if dim is not None else int(np.asarray(x0).shape[0])
    lower = jnp.zeros(d, dtype=jnp.float64)
    upper = jnp.ones(d, dtype=jnp.float64)
    if neg_acq is None:
        neg_acq = _get_jitted_neg_acq(acq.round_info)
    solver = LBFGSB(fun=neg_acq, maxiter=200)
    result = solver.run(
        jnp.asarray(x0, dtype=jnp.float64),
        (lower, upper),
        *extra_args,
        acq.gp_mu_state, acq.gp_log_var_state,
        acq.scaling, acq._g_best, acq.jitter,
    )
    x_opt = np.clip(np.asarray(result.params), 0.0, 1.0)
    val = float(-result.state.value)
    if not np.isfinite(val):
        logger.warning("lbfgsb_acquisition_non_finite", val=val)
    return x_opt, val


def _sobol_screen(acq, dim, seed, pow_sobol, n_restarts, acq_batch=None,
                  extra_acq_args=()):
    """Return (top_candidates, top_acq_values) sorted ascending by acq value.
    extra_acq_args is forwarded to acq_batch before the GP/scaling state.
    When acq_batch is None, defaults to the round_info-aware batched acq."""
    jnp = _jnp()
    sampler = Sobol(d=dim, scramble=True, seed=seed)
    candidates = sampler.random(2 ** pow_sobol)
    if acq_batch is None:
        acq_batch = _get_jitted_acq_batch(acq.round_info)
    aqn_vals = np.asarray(acq_batch(
        jnp.asarray(candidates, dtype=jnp.float64),
        *extra_acq_args,
        acq.gp_mu_state, acq.gp_log_var_state,
        acq.scaling, acq._g_best, acq.jitter,
    ))
    top_idx = np.argsort(aqn_vals)[-n_restarts:]
    return candidates[top_idx], aqn_vals[top_idx]


def _branch_no_bound_optimise_acquisition(
    acq: "ExpectedImprovement",
    cat_vars: list[tuple[int, list[float]]],
    seed: int,
    pow_sobol: int,
    n_restarts: int,
) -> tuple[np.ndarray, float]:
    """Branch (no bound) over integer/categorical dims; LBFGS-B on the rest.

    Falls back to the best raw Sobol candidate if LBFGS-B refinement returns
    only non-finite values across every (combo, restart) pair — happens when
    the surrogate is degenerate (e.g. all points contaminated by outliers).

    The masked acq functions are JIT'd ONCE per index pattern (cached on
    _JIT_MASKED_*_BY_MASK) and reused across every combo by passing
    mask_values as a JAX array argument. Without this, each combo
    recompiles a fresh XLA program (n_combos × per-iter cache clear)."""
    from itertools import product

    D_full = int(acq.gp_mu_state["X"].shape[1])

    # No categorical dims → plain Sobol screen + LBFGS-B over the full unit cube.
    if not cat_vars:
        starts, start_vals = _sobol_screen(
            acq, dim=D_full, seed=seed, pow_sobol=pow_sobol, n_restarts=n_restarts,
        )
        best_x, best_val = None, -np.inf
        for x0 in starts:
            x_opt, val = _lbfgsb_maximise_acquisition(acq, x0)
            if np.isfinite(val) and val > best_val:
                best_x, best_val = x_opt, val
        if best_x is None:
            best_idx = int(np.argmax(start_vals))
            return np.asarray(starts[best_idx], dtype=float), float(start_vals[best_idx])
        return best_x, best_val

    # Sort indices once so the cache key is canonical and the in-trace
    # left/right slicing is monotonic.
    raw_indices, raw_values = zip(*cat_vars)
    order = sorted(range(len(raw_indices)), key=lambda i: raw_indices[i])
    cat_indices = tuple(int(raw_indices[i]) for i in order)
    cat_values = [list(raw_values[i]) for i in order]

    combinations = list(product(*cat_values))
    D_red = D_full - len(cat_indices)
    cat_set = set(cat_indices)
    cont_dims = [i for i in range(D_full) if i not in cat_set]
    cat_idx_list = list(cat_indices)

    masked_neg_acq = _get_jitted_masked_neg_acq(cat_indices, acq.round_info)
    masked_acq_batch = _get_jitted_masked_acq_batch(cat_indices, acq.round_info)
    jnp = _jnp()

    best_x_full, best_val = None, -np.inf
    fallback_x_full, fallback_val = None, -np.inf

    for combo in combinations:
        mask_values = jnp.asarray(combo, dtype=jnp.float64)
        combo_arr = np.asarray(combo, dtype=float)

        starts, start_vals = _sobol_screen(
            acq, dim=D_red, seed=seed, pow_sobol=pow_sobol,
            n_restarts=n_restarts, acq_batch=masked_acq_batch,
            extra_acq_args=(mask_values,),
        )

        # Track the best raw Sobol candidate across all combos as a fallback.
        best_start_idx = int(np.argmax(start_vals))
        sv = float(start_vals[best_start_idx])
        if np.isfinite(sv) and sv > fallback_val:
            x_full = np.empty(D_full)
            x_full[cat_idx_list] = combo_arr
            x_full[cont_dims] = np.asarray(starts[best_start_idx])
            fallback_x_full, fallback_val = x_full, sv

        combo_best_lbfgs = -np.inf
        n_finite_lbfgs = 0
        for x0 in starts:
            x_opt_red, val = _lbfgsb_maximise_acquisition(
                acq, x0, neg_acq=masked_neg_acq, dim=D_red,
                extra_args=(mask_values,),
            )
            if np.isfinite(val):
                n_finite_lbfgs += 1
                if val > combo_best_lbfgs:
                    combo_best_lbfgs = val
                if val > best_val:
                    x_full = np.empty(D_full)
                    x_full[cat_idx_list] = combo_arr
                    x_full[cont_dims] = x_opt_red
                    best_x_full, best_val = x_full, val

        logger.debug(
            "branch_no_bound_combo",
            combo_unit=list(combo_arr),
            sobol_top_acq=sv,
            lbfgs_best_acq=float(combo_best_lbfgs) if np.isfinite(combo_best_lbfgs) else None,
            n_finite_lbfgs=n_finite_lbfgs,
            n_starts=len(starts),
        )

    if best_x_full is None:
        if fallback_x_full is None:
            raise RuntimeError(
                "Branch-no-bound acquisition produced no finite values across "
                f"{len(combinations)} combos — surrogate likely degenerate."
            )
        logger.warning("branch_no_bound_lbfgs_all_failed",
                       n_combos=len(combinations),
                       fallback_acq=fallback_val)
        return fallback_x_full, fallback_val

    return best_x_full, best_val