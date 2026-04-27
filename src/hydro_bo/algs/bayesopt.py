"""
Bayesian Optimiser with heteroscedastic noise modelling and noisy EI.

f(x) returns a 1D iterable of sample evaluations (variable length;
NaN/Inf entries are dropped). Per BO iteration two GPs are trained:

  - log-variance GP on (x_i, log s^2_i) with per-point noise ~ 2/(n_i-1),
  - mean GP on (x_i, mu_i) with per-point noise sigma^2(x_i)/n_i, where
    sigma^2(x_i) is the log-variance GP's prediction at x_i.

The latent objective is g(x) = mu(x) - lam * sigma(x), where
sigma(x) = sqrt(sigma^2(x)) keeps lam in the same units as f. Its
posterior distribution is approximated as a moment-matched Gaussian
using independence of the two GPs and log-normal moments of sigma(x)
= exp(log_sigma2(x) / 2). Noisy expected improvement uses
g_best = max_i E[g(x_i)] over training points as the incumbent.

X is scaled to the unit cube via fixed bounds; targets are z-scored.
The acquisition is screened on a Sobol grid in [0,1]^d and refined
with LBFGS-B using JAX autodiff gradients.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Sequence

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
from scipy.stats.qmc import Sobol

from hydro_bo.algs.logging_config import get_logger

logger = get_logger(__name__)


def configure_jax_threads(n_threads: int) -> None:
    """Set XLA CPU intra/inter-op thread counts. Call BEFORE the first jax
    import so the BLAS pool inside Cholesky / matmul uses the requested
    cores. Safe to call multiple times before jax is imported; a no-op
    warning is logged if jax has already initialised."""
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


def _jax():
    import jax
    return jax


def _jnp():
    import jax.numpy as jnp
    return jnp


@dataclass
class Dataset:
    """Stores ragged sample evaluations and exposes per-point summary
    statistics, target standardisation, and unit-cube X scaling.

    Each row of _X has an associated 1D array of f(x) samples; arrays may
    have different lengths (failed runs dropped before storage).
    """
    _X: np.ndarray
    _samples: list
    bounds: np.ndarray

    def __post_init__(self):
        b = np.asarray(self.bounds, dtype=float)
        self._lo = b[:, 0]
        self._span = b[:, 1] - b[:, 0]

        samples = [np.asarray(s, dtype=float).ravel() for s in self._samples]
        self._samples = samples
        self.n = np.array([len(s) for s in samples], dtype=int)

        with np.errstate(invalid="ignore"):
            self.mu = np.array([
                float(np.mean(s)) if len(s) >= 1 else np.nan for s in samples
            ])
            # Unbiased sample variance estimating the population variance.
            self.sigma2 = np.array([
                float(np.var(s, ddof=1)) if len(s) >= 2 else np.nan
                for s in samples
            ])

        floor = 1e-12
        log_v = np.full_like(self.sigma2, np.nan)
        valid_v = np.isfinite(self.sigma2) & (self.sigma2 > 0)
        log_v[valid_v] = np.log(np.maximum(self.sigma2[valid_v], floor))
        self.log_sigma2 = log_v

        finite_mu = self.mu[np.isfinite(self.mu)]
        finite_lv = self.log_sigma2[np.isfinite(self.log_sigma2)]
        self._mu_y = float(np.mean(finite_mu)) if finite_mu.size else 0.0
        sy = float(np.std(finite_mu)) if finite_mu.size else 1.0
        self._sigma_y = sy if sy > 0 else 1.0
        self._mu_lv = float(np.mean(finite_lv)) if finite_lv.size else 0.0
        slv = float(np.std(finite_lv)) if finite_lv.size else 1.0
        self._sigma_lv = slv if slv > 0 else 1.0

    @property
    def X(self):
        return self._X

    @property
    def X_scaled(self):
        return (self._X - self._lo) / self._span

    def to_unit(self, X: np.ndarray) -> np.ndarray:
        return (X - self._lo) / self._span

    def to_original(self, X_unit: np.ndarray) -> np.ndarray:
        return self._lo + X_unit * self._span

    @property
    def mu_scaled(self):
        return (self.mu - self._mu_y) / self._sigma_y

    @property
    def log_sigma2_scaled(self):
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

class BaseGP(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, noise: np.ndarray) -> None:
        """Fit the GP to data (X, y) with per-point noise variances."""
        ...
    
    @abstractmethod
    def predict(self, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict (mean, var) at X_test."""
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

    @staticmethod
    def _kernel(log_amp, log_ls, X1, X2):
        jnp = _jnp()
        amp2 = jnp.exp(2.0 * log_amp)
        ls = jnp.exp(log_ls)
        diff = (X1[:, None, :] - X2[None, :, :]) / ls
        d2 = jnp.sum(diff * diff, axis=-1)
        return amp2 * jnp.exp(-0.5 * d2)

    @staticmethod
    def _neg_mll_static(params, X, y, noise, jitter):
        jnp = _jnp()
        jax = _jax()
        n = X.shape[0]
        K = HeteroscedasticGP._kernel(params["log_amp"], params["log_ls"], X, X)
        K = K + jnp.diag(noise) + jitter * jnp.eye(n)
        y_c = y - params["mean"]
        L = jnp.linalg.cholesky(K)
        alpha = jax.scipy.linalg.cho_solve((L, True), y_c)
        log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
        return 0.5 * (jnp.dot(y_c, alpha) + log_det + n * jnp.log(2.0 * jnp.pi))

    @staticmethod
    def _factorise_static(params, X, y, noise, jitter):
        jnp = _jnp()
        jax = _jax()
        n = X.shape[0]
        K = HeteroscedasticGP._kernel(params["log_amp"], params["log_ls"], X, X)
        K = K + jnp.diag(noise) + jitter * jnp.eye(n)
        L = jnp.linalg.cholesky(K)
        alpha = jax.scipy.linalg.cho_solve((L, True), y - params["mean"])
        return L, alpha

    @staticmethod
    def _predict_static(params, X_train, L, alpha, mean, X_test):
        jnp = _jnp()
        jax = _jax()
        K_s = HeteroscedasticGP._kernel(params["log_amp"], params["log_ls"], X_test, X_train)
        amp2 = jnp.exp(2.0 * params["log_amp"])
        mu = mean + K_s @ alpha
        v = jax.scipy.linalg.cho_solve((L, True), K_s.T)
        var = amp2 - jnp.sum(K_s * v.T, axis=1)
        var = jnp.clip(var, 1e-12, None)
        return mu, var

    def fit(self, X, y, noise):
        from jaxopt import LBFGS
        jnp = _jnp()
        X = jnp.asarray(X, dtype=jnp.float64)
        y = jnp.asarray(y, dtype=jnp.float64).reshape(-1)
        noise = jnp.asarray(noise, dtype=jnp.float64).reshape(-1)
        n, d = int(X.shape[0]), int(X.shape[1])

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
        nmll = _get_jitted_nmll()
        solver = LBFGS(fun=nmll, maxiter=self.max_iters)
        result = solver.run(init, X, y, noise, jitter)
        self._release_buffers()
        self.params = {k: jnp.asarray(v) for k, v in result.params.items()}

        L, alpha = _get_jitted_factorise()(self.params, X, y, noise, jitter)
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
        """Returns (mean, var) at X_test in standardised target space."""
        jnp = _jnp()
        X_test = jnp.asarray(X_test, dtype=jnp.float64)
        if X_test.ndim == 1:
            X_test = X_test[None, :]
        return _get_jitted_predict()(
            self.params, self._X, self._L, self._alpha, self._mean, X_test
        )


# Module-level jit handles. 
_JIT_NMLL = None
_JIT_FACTORISE = None
_JIT_PREDICT = None
_JIT_ACQ_G_STATS = None
_JIT_ACQ_EVAL = None
_JIT_ACQ_BATCH = None
_JIT_NEG_ACQ = None


def _get_jitted_nmll():
    global _JIT_NMLL
    if _JIT_NMLL is None:
        jax = _jax()
        _JIT_NMLL = jax.jit(HeteroscedasticGP._neg_mll_static)
    return _JIT_NMLL


def _get_jitted_factorise():
    global _JIT_FACTORISE
    if _JIT_FACTORISE is None:
        jax = _jax()
        _JIT_FACTORISE = jax.jit(HeteroscedasticGP._factorise_static)
    return _JIT_FACTORISE


def _get_jitted_predict():
    global _JIT_PREDICT
    if _JIT_PREDICT is None:
        jax = _jax()
        _JIT_PREDICT = jax.jit(HeteroscedasticGP._predict_static)
    return _JIT_PREDICT


def _acq_g_stats_pure(x, gp_mu_state, gp_log_var_state, scaling):
    """Pure version of NoisyEI._g_stats — all closure state passed explicitly
    so a single module-level jit can serve every BO iteration."""
    jnp = _jnp()
    mu_y, sigma_y, mu_lv, sigma_lv, lam = scaling
    mu_s, var_mu_s = HeteroscedasticGP._predict_static(
        gp_mu_state["params"], gp_mu_state["X"], gp_mu_state["L"],
        gp_mu_state["alpha"], gp_mu_state["mean"], x,
    )
    lv_s, var_lv_s = HeteroscedasticGP._predict_static(
        gp_log_var_state["params"], gp_log_var_state["X"], gp_log_var_state["L"],
        gp_log_var_state["alpha"], gp_log_var_state["mean"], x,
    )
    m_mu = mu_s * sigma_y + mu_y
    v_mu = var_mu_s * (sigma_y ** 2)
    m_lv = lv_s * sigma_lv + mu_lv
    v_lv = var_lv_s * (sigma_lv ** 2)
    e_sd = jnp.exp(0.5 * m_lv + 0.125 * v_lv)
    var_sd = (jnp.exp(0.25 * v_lv) - 1.0) * jnp.exp(m_lv + 0.25 * v_lv)
    e_g = m_mu - lam * e_sd
    var_g = v_mu + (lam ** 2) * var_sd
    return e_g, var_g


def _acq_eval_pure(x, gp_mu_state, gp_log_var_state, scaling, g_best, jitter):
    """Noisy EI value at a single point. x has shape (d,)."""
    jnp = _jnp()
    from jax.scipy.stats.norm import cdf, pdf
    x2 = x[None, :]
    e_g, var_g = _acq_g_stats_pure(x2, gp_mu_state, gp_log_var_state, scaling)
    sigma_g = jnp.sqrt(jnp.clip(var_g, jitter, None))
    delta = e_g - g_best
    z = delta / sigma_g
    ei = delta * cdf(z) + sigma_g * pdf(z)
    return ei.squeeze()


def _neg_acq_pure(x, gp_mu_state, gp_log_var_state, scaling, g_best, jitter):
    return -_acq_eval_pure(x, gp_mu_state, gp_log_var_state, scaling, g_best, jitter)


def _get_jitted_acq_g_stats():
    global _JIT_ACQ_G_STATS
    if _JIT_ACQ_G_STATS is None:
        jax = _jax()
        _JIT_ACQ_G_STATS = jax.jit(_acq_g_stats_pure)
    return _JIT_ACQ_G_STATS


def _get_jitted_acq_eval():
    global _JIT_ACQ_EVAL
    if _JIT_ACQ_EVAL is None:
        jax = _jax()
        _JIT_ACQ_EVAL = jax.jit(_acq_eval_pure)
    return _JIT_ACQ_EVAL


def _get_jitted_acq_batch():
    global _JIT_ACQ_BATCH
    if _JIT_ACQ_BATCH is None:
        jax = _jax()
        _JIT_ACQ_BATCH = jax.jit(jax.vmap(
            _acq_eval_pure, in_axes=(0, None, None, None, None, None)
        ))
    return _JIT_ACQ_BATCH


def _get_jitted_neg_acq():
    global _JIT_NEG_ACQ
    if _JIT_NEG_ACQ is None:
        jax = _jax()
        _JIT_NEG_ACQ = jax.jit(_neg_acq_pure)
    return _JIT_NEG_ACQ


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


class NoisyExpectedImprovement(AcquisitionFunction):
    """
    Approximate EI for a heteroscedastic GP objective. Our objective is not 
    purely normal as g(x) = mu(x) - lam * sigma(x) with sigma(x) ~ LogNormal.

    Due to the scale difference of mu and signa, we assume that g(x) is still 
    normally distributed, and match first and second order moments. (Assuming 
    independenceof mu and sigma).

    The acquisition is then EI with respect to the incumbent g_best = max_i E[g(x_i)].

    g_best = max(mu_i - lam * sigma_i)
    """

    def __init__(self, gp_mu: HeteroscedasticGP, gp_log_var: HeteroscedasticGP,
                 lam: float, dataset: Dataset, jitter: float = 1e-9):
        jnp = _jnp()
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
        e_g, _ = _get_jitted_acq_g_stats()(
            X_unit, self.gp_mu_state, self.gp_log_var_state, self.scaling
        )
        return jnp.max(e_g)

    def evaluate(self, x):
        return _get_jitted_acq_eval()(
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
    ):
        self.f = f
        self.bounds = np.asarray(bounds, dtype=float)
        self.n_initial_points = n_initial_points
        self.iter_limit = iter_limit
        self.lam = lam
        self.n_restarts = n_restarts
        self.pow_sobol = pow_sobol
        self.seed = seed

        self.gp_mu = HeteroscedasticGP()
        self.gp_log_var = HeteroscedasticGP()
        self._X: list[np.ndarray] = []
        self._samples: list[np.ndarray] = []
        self.dataset: Dataset | None = None

    def observe(self, x: np.ndarray, samples) -> None:
        """Record an observation: x in original bounds, samples = 1D iterable."""
        x = np.asarray(x, dtype=float)
        s = np.asarray(samples, dtype=float).ravel()
        s = s[np.isfinite(s)]
        self._X.append(x)
        self._samples.append(s)

    def _evaluate_and_store(self, x: np.ndarray) -> np.ndarray:
        s = np.asarray(self.f(x), dtype=float).ravel()
        s = s[np.isfinite(s)]
        self._X.append(np.asarray(x, dtype=float))
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

    def _fit_gps(self) -> None:
        jnp = _jnp()
        X = np.stack(self._X)
        self.dataset = Dataset(X, self._samples, self.bounds)
        ds = self.dataset
        X_unit = ds.X_scaled

        # 1) Log-variance GP on rows with n >= 2.
        m_lv = ds.mask_log_sigma2
        log_var_fit = False
        if int(m_lv.sum()) >= 2:
            X_lv = X_unit[m_lv]
            y_lv = ds.log_sigma2_scaled[m_lv]
            noise_lv = ds.noise_log_sigma2[m_lv] / (ds._sigma_lv ** 2)
            self.gp_log_var.fit(X_lv, y_lv, noise_lv)
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
        self.gp_mu.fit(X_mu, y_mu, noise_mu_scaled)
        logger.info("bo_mean_gp_fit", n_points=int(m_mu.sum()),
                    used_log_var_gp=log_var_fit)

    def _suggest(self) -> np.ndarray:
        if self.dataset is None:
            self._fit_gps()
        logger.debug("bo_acquisition_optimise",
                     lam=self.lam, n_restarts=self.n_restarts)
        acq = NoisyExpectedImprovement(
            self.gp_mu, self.gp_log_var, self.lam, self.dataset
        )
        logger.debug("bo_noisy_ei_incumbent", g_best=float(acq._g_best))

        starts_unit = self._sobol_screen(acq)

        best_x_unit, best_val = None, -np.inf
        for x0 in starts_unit:
            x_opt, val = _lbfgsb_maximise_acquisition(acq, x0)
            if val > best_val:
                best_x_unit, best_val = x_opt, val

        logger.debug("bo_acquisition_optimise_complete", best_score=float(best_val))
        x_orig = self.dataset.to_original(np.asarray(best_x_unit))

        # Drop the per-iteration JAX device buffers and compiled XLA programs
        # so the driver's RSS doesn't grow with each BO step.
        del acq
        _clear_jax_caches()
        import gc
        gc.collect()

        return x_orig

    def _sobol_screen(self, acq: "NoisyExpectedImprovement") -> np.ndarray:
        """Evaluate acquisition on Sobol points in [0,1]^d and return top n_restarts."""
        jnp = _jnp()
        d = self.bounds.shape[0]
        sampler = Sobol(d=d, scramble=True, seed=self.seed + len(self._X))
        candidates = sampler.random(2 ** self.pow_sobol)
        acq_batch = _get_jitted_acq_batch()
        aqn_vals = np.asarray(acq_batch(
            jnp.asarray(candidates, dtype=jnp.float64),
            acq.gp_mu_state, acq.gp_log_var_state,
            acq.scaling, acq._g_best, acq.jitter,
        ))
        top_idx = np.argsort(aqn_vals)[-self.n_restarts:]
        return candidates[top_idx]


def _lbfgsb_maximise_acquisition(
    acq: "NoisyExpectedImprovement",
    x0: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Maximise acq over the unit cube using jaxopt LBFGS-B with autodiff.

    Uses the module-level jitted neg-acq via solver.run extra args so no fresh
    closure / jit is created per call."""
    from jaxopt import LBFGSB
    jnp = _jnp()

    d = int(np.asarray(x0).shape[0])
    lower = jnp.zeros(d, dtype=jnp.float64)
    upper = jnp.ones(d, dtype=jnp.float64)

    neg_acq = _get_jitted_neg_acq()
    solver = LBFGSB(fun=neg_acq, maxiter=200)
    result = solver.run(
        jnp.asarray(x0, dtype=jnp.float64),
        (lower, upper),
        acq.gp_mu_state,
        acq.gp_log_var_state,
        acq.scaling,
        acq._g_best,
        acq.jitter,
    )
    x_opt = np.clip(np.asarray(result.params), 0.0, 1.0)
    val = float(-result.state.value)
    if not np.isfinite(val):
        logger.warning("lbfgsb_acquisition_non_finite", val=val)
    return x_opt, val
