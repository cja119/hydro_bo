"""GP surrogates used by the Bayesian optimiser."""

from abc import ABC, abstractmethod
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp


def _project_integer_dims(X, round_info):
    """Snap the integer dims of X to their nearest unit-cube grid position.
    round_info: sorted tuple of (dim_idx, n_levels) — both static Python ints.
    Empty round_info is a no-op (returns X unchanged)."""
    if not round_info:
        return X
    for idx, n in round_info:
        if n <= 1:
            continue
        denom = n - 1
        col = X[..., idx]
        rounded = jnp.round(col * denom) / denom
        X = X.at[..., idx].set(rounded)
    return X


@partial(jax.jit, static_argnames="round_info")
def _kernel(log_amp, log_ls, X1, X2, round_info):
    """ARD RBF kernel with optional projection of integer dims onto a
    discrete unit-cube grid. round_info is a static Python tuple — empty
    skips projection."""
    if round_info:
        X1 = _project_integer_dims(X1, round_info)
        X2 = _project_integer_dims(X2, round_info)
    amp2 = jnp.exp(2.0 * log_amp)
    ls = jnp.exp(log_ls)
    diff = (X1[:, None, :] - X2[None, :, :]) / ls
    d2 = jnp.sum(diff * diff, axis=-1)
    return amp2 * jnp.exp(-0.5 * d2)


@partial(jax.jit, static_argnames="round_info")
def _neg_mll(params, X, y, noise, jitter, round_info):
    n = X.shape[0]
    K = _kernel(params["log_amp"], params["log_ls"], X, X, round_info)
    K = K + jnp.diag(noise) + jitter * jnp.eye(n)
    y_c = y - params["mean"]
    L = jnp.linalg.cholesky(K)
    alpha = jax.scipy.linalg.cho_solve((L, True), y_c)
    log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    return 0.5 * (jnp.dot(y_c, alpha) + log_det + n * jnp.log(2.0 * jnp.pi))


@partial(jax.jit, static_argnames="round_info")
def _factorise(params, X, y, noise, jitter, round_info):
    n = X.shape[0]
    K = _kernel(params["log_amp"], params["log_ls"], X, X, round_info)
    K = K + jnp.diag(noise) + jitter * jnp.eye(n)
    L = jnp.linalg.cholesky(K)
    alpha = jax.scipy.linalg.cho_solve((L, True), y - params["mean"])
    return L, alpha


@partial(jax.jit, static_argnames="round_info")
def _predict(params, X_train, L, alpha, mean, X_test, round_info):
    K_s = _kernel(params["log_amp"], params["log_ls"], X_test, X_train, round_info)
    amp2 = jnp.exp(2.0 * params["log_amp"])
    mu = mean + K_s @ alpha
    v = jax.scipy.linalg.cho_solve((L, True), K_s.T)
    var = amp2 - jnp.sum(K_s * v.T, axis=1)
    var = jnp.clip(var, 1e-12, None)
    return mu, var

@partial(jax.jit, static_argnames="round_info")
def _predict_zero_mean(params, X_train, L, alpha, X_test, round_info):
    params_zero_mean = {**params, "mean": jnp.array(0.0, dtype=jnp.float64)}
    return _predict(
        params_zero_mean,
        X_train,
        L,
        alpha,
        params_zero_mean["mean"],
        X_test,
        round_info,
    )


@partial(jax.jit, static_argnames=("round_info", "vi_max_iters", "vi_tol"))
def _vi_inner_loop(params, X, k, N, jitter, round_info, vi_max_iters, vi_tol):
    """PG-VI alternating updates. Returns (omega, m, S_diag) at convergence.

    Iterates with `lax.while_loop` (no autodiff trace, O(1) memory) and
    early-stops once max|Δω| ≤ vi_tol. The output is wrapped in
    `stop_gradient`: the (ω, m, S) updates are the closed-form ELBO
    maximisers in their respective variables, so at the joint fixed point
    ∂ELBO/∂(ω, m, S) = 0. By the envelope theorem, ∇_θ ELBO equals the
    explicit partial derivative through K(θ) alone — gradients through the
    inner loop are zero at convergence and can be safely cut.
    """
    n = X.shape[0]

    K = _kernel(params["log_amp"], params["log_ls"], X, X, round_info)
    K = K + jitter * jnp.eye(n)
    kappa = k - N / 2.0

    def step(omega):
        noise = 1.0 / omega
        L = jnp.linalg.cholesky(K + jnp.diag(noise))
        alpha = jax.scipy.linalg.cho_solve((L, True), kappa / omega)
        m = K @ alpha
        v = jax.scipy.linalg.cho_solve((L, True), K)
        S_diag = jnp.clip(jnp.diag(K) - jnp.sum(K * v.T, axis=1), 1e-12, None)
        c = jnp.maximum(jnp.sqrt(m**2 + S_diag), 1e-6)
        omega_new = N / (2.0 * c) * jnp.tanh(c / 2.0)
        return omega_new, m, S_diag

    omega0 = jnp.broadcast_to(N / 4.0, (n,))
    omega1, _, _ = step(omega0)

    def cond_fun(carry):
        i, omega, omega_prev = carry
        return jnp.logical_and(
            i < vi_max_iters,
            jnp.max(jnp.abs(omega - omega_prev)) > vi_tol,
        )

    def body_fun(carry):
        i, omega, _ = carry
        omega_new, _, _ = step(omega)
        return i + 1, omega_new, omega

    _, omega_star, _ = jax.lax.while_loop(
        cond_fun, body_fun, (jnp.int32(1), omega1, omega0)
    )
    _, m_star, S_diag_star = step(omega_star)

    return (
        jax.lax.stop_gradient(omega_star),
        jax.lax.stop_gradient(m_star),
        jax.lax.stop_gradient(S_diag_star),
    )


@partial(jax.jit, static_argnames=("round_info", "vi_max_iters", "vi_tol"))
def _neg_elbo(params, X, k, N, jitter, round_info, vi_max_iters, vi_tol):
    """Negative ELBO for PG-augmented Binomial GP.

    Runs the inner PG-VI fixed-point loop to convergence at the given
    kernel hyperparameters, then evaluates the variational lower bound
    on log p(k | X, hyperparameters).

    Args:
        params: dict with "log_amp", "log_ls" (kernel hyperparameters).
            Note: no "mean" — latent f is zero-mean by convention.
        X: (n, d) training inputs.
        k: (n,) success counts.
        N: (n,) trial counts.
        jitter: scalar diagonal regularisation.
        round_info: static tuple of integer-dimension info (for kernel).
        vi_max_iters: int, number of inner VI iterations.

    Returns:
        Scalar negative ELBO. Suitable for L-BFGS minimisation.
    """
    n = X.shape[0]

    # 1. Run inner VI loop to converged (omega, m, S_diag).
    omega, m, S_diag = _vi_inner_loop(
        params, X, k, N, jitter, round_info, vi_max_iters, vi_tol
    )

    # 2. Build the kernel matrix at converged hyperparameters.
    K = _kernel(params["log_amp"], params["log_ls"], X, X, round_info)
    K = K + jitter * jnp.eye(n)

    # 3. Convenience quantities.
    kappa = k - N / 2.0  # centred sufficient statistic
    c = jnp.sqrt(m**2 + S_diag)  # PG tilting parameter
    noise = 1.0 / omega  # synthetic noise variance

    # 4. Cholesky factorisations we need.
    # K_plus_noise = K + diag(1/ω) — used for the variational covariance.
    K_plus_noise = K + jnp.diag(noise)
    L_kn = jnp.linalg.cholesky(K_plus_noise)
    # L_K = Cholesky of K — used for the prior KL term.
    L_K = jnp.linalg.cholesky(K)

    # 5. ELBO terms.

    # 5a. Expected log-likelihood under q.
    #     E_q[log p(k|N,f,ω)] = κ^T m - 1/2 sum_i ω_i (m_i^2 + S_ii)
    #                        + sum_i [- N_i log 2  ... constants ...]
    # Constants don't depend on hyperparameters, drop them for L-BFGS.
    expected_loglik = jnp.dot(kappa, m) - 0.5 * jnp.sum(omega * (m**2 + S_diag))

    # 5b. Cross-entropy / PG term from E_q[log p(ω) / q(ω)].
    #     For PG(N, 0) prior and PG(N, c) posterior, this works out to
    #     sum_i [N_i log cosh(c_i / 2) - c_i^2 ω_i / 2]
    pg_kl = jnp.sum(N * jnp.log(jnp.cosh(c / 2.0)) - 0.5 * c**2 * omega)

    # 5c. Gaussian-Gaussian KL: KL(q(f) || p(f)) where p(f) = N(0, K),
    #     q(f) = N(m, S) with S = (K^{-1} + diag(ω))^{-1}.
    #
    #     KL = 1/2 [tr(K^{-1} S) + m^T K^{-1} m - n + log|K| - log|S|]

    # tr(K^{-1} S): use the identity K^{-1} S = I - diag(ω) S, so
    #              tr(K^{-1} S) = n - tr(diag(ω) S) = n - sum_i ω_i S_ii
    trace_K_inv_S = n - jnp.sum(omega * S_diag)

    # m^T K^{-1} m via Cholesky of K
    K_inv_m = jax.scipy.linalg.cho_solve((L_K, True), m)
    quad_form = jnp.dot(m, K_inv_m)

    # log|K| from L_K
    log_det_K = 2.0 * jnp.sum(jnp.log(jnp.diag(L_K)))

    # log|S| via the identity |S| = |K| / (|K + 1/ω| * prod(ω))
    #   derivation: S^{-1} = K^{-1} + diag(ω); so |S^{-1}| = |K^{-1}| |I + diag(ω) K|
    #   ... actually cleaner via Sylvester: |K^{-1} + diag(ω)| = |K|^{-1} |K + diag(1/ω)| prod(ω)
    log_det_S = (
        log_det_K - 2.0 * jnp.sum(jnp.log(jnp.diag(L_kn))) - jnp.sum(jnp.log(omega))
    )

    gaussian_kl = 0.5 * (trace_K_inv_S + quad_form - n + log_det_K - log_det_S)

    # 6. Assemble ELBO.
    #    L = E_q[log p(k|f,ω)] - KL(q(f)||p(f)) - KL(q(ω)||p(ω))
    #    where the PG KL is what we called `pg_kl`.
    elbo = expected_loglik - gaussian_kl - pg_kl

    return -elbo


class BaseGP(ABC):
    @abstractmethod
    def fit(
        self, X: np.ndarray, y: np.ndarray, noise: np.ndarray, round_info: tuple = ()
    ) -> None:
        """Fit the GP to data (X, y) with per-point noise variances."""
        ...

    @abstractmethod
    def predict(self, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict (mean, var) at X_test using the round_info from fit time."""
        ...

    @abstractmethod
    def state(self) -> dict:
        """Snapshot of the fitted state needed by acquisition functions."""
        ...

    def _release_buffers(self):
        for attr in ("_X", "_L", "_alpha", "_mean"):
            buf = getattr(self, attr, None)
            if buf is not None and hasattr(buf, "delete"):
                try:
                    buf.delete()
                except Exception:
                    pass
            setattr(self, attr, None)


class HeteroscedasticGP(BaseGP):
    """Exact GP regression with constant mean, ARD RBF kernel, and fixed
    per-point observation noise. Hyperparameters are fit via the shared
    multistart-SQP helper (`hydro_bo.opt.solvers.multistart_sqp`) — Sobol
    screen → top-K SQP refines → best converged."""

    def __init__(
        self,
        jitter: float = 1e-6,
        pow_sobol_fit: int = 10,
        n_restarts_fit: int = 8,
        sqp_config=None,
        seed: int = 0,
    ):
        self.jitter = jitter
        self.pow_sobol_fit = int(pow_sobol_fit)
        self.n_restarts_fit = int(n_restarts_fit)
        self.sqp_config = sqp_config
        self.seed = int(seed)
        self.params = None
        self._X = None
        self._L = None
        self._alpha = None
        self._mean = None
        self.round_info: tuple = ()

    def state(self) -> dict:
        """Snapshot of the fitted state needed by the acquisition."""
        return {
            "params": self.params,
            "X": self._X,
            "L": self._L,
            "alpha": self._alpha,
            "mean": self._mean,
        }

    def fit(self, X, y, noise, round_info: tuple = ()):
        from hydro_bo.opt.solvers import multistart_sqp
        from hydro_bo.utils.logging_config import get_logger

        logger = get_logger(__name__)

        X = jnp.asarray(X, dtype=jnp.float64)
        y = jnp.asarray(y, dtype=jnp.float64).reshape(-1)
        noise = jnp.asarray(noise, dtype=jnp.float64).reshape(-1)
        n, d = int(X.shape[0]), int(X.shape[1])
        round_info = tuple(round_info)
        self.round_info = round_info
        jitter = jnp.asarray(self.jitter, dtype=jnp.float64)

        logger.debug(
            "gp_fit_start", n_datapoints=n, n_dims=d, warm_start=self.params is not None
        )

        # Decision vector: [log_amp, log_ls (d), mean] — flattened for the
        # multistart-SQP helper. Bounds: log-amp/ls in a generous log-space
        # box, mean in a 3σ band around the data mean.
        y_mean = float(np.mean(np.asarray(y)))
        y_std = float(np.std(np.asarray(y))) or 1.0
        lb = np.concatenate([[-5.0], np.full(d, -5.0), [y_mean - 3.0 * y_std]])
        ub = np.concatenate([[5.0], np.full(d, 5.0), [y_mean + 3.0 * y_std]])

        # SQP minimises: NMLL is already a loss, no negation needed.
        def _nmll_obj(x_vec, _p):
            params = {
                "log_amp": x_vec[0],
                "log_ls": jax.lax.dynamic_slice(x_vec, (1,), (d,)),
                "mean": x_vec[1 + d],
            }
            return _neg_mll(params, X, y, noise, jitter, round_info).reshape(())

        x_best, val_best, info = multistart_sqp(
            _nmll_obj,
            (lb, ub),
            seed=self.seed + len(round_info),  # vary across calls
            pow_sobol=self.pow_sobol_fit,
            n_restarts=self.n_restarts_fit,
            sqp_config=self.sqp_config,
        )

        self._release_buffers()
        self.params = {
            "log_amp": jnp.asarray(x_best[0], dtype=jnp.float64),
            "log_ls": jnp.asarray(x_best[1 : 1 + d], dtype=jnp.float64),
            "mean": jnp.asarray(x_best[1 + d], dtype=jnp.float64),
        }
        L, alpha = _factorise(self.params, X, y, noise, jitter, round_info)
        self._X = X
        self._L = L
        self._alpha = alpha
        self._mean = self.params["mean"]

        logger.debug(
            "gp_fit_complete",
            final_nmll=val_best,
            n_starts=info["n_starts"],
            n_converged=info.get("n_converged", 0),
            converged=info["converged"],
        )

    def predict(self, X_test):
        """Returns (mean, var) at X_test in standardised target space.
        Uses self.round_info captured at fit time so predictions are
        kernel-consistent with the fit."""
        X_test = jnp.asarray(X_test, dtype=jnp.float64)
        if X_test.ndim == 1:
            X_test = X_test[None, :]
        return _predict(
            self.params,
            self._X,
            self._L,
            self._alpha,
            self._mean,
            X_test,
            self.round_info,
        )


class BinomialGP(BaseGP):
    """Polya-Gamma augmented GP for binary / count feasibility data.

    Outer minimisation of the negative ELBO uses the shared multistart-SQP
    helper; inner Polya-Gamma VI is unchanged."""

    def __init__(
        self,
        jitter: float = 1e-6,
        vi_max_iters: int = 30,
        vi_tol: float = 1e-4,
        pow_sobol_fit: int = 10,
        n_restarts_fit: int = 8,
        sqp_config=None,
        seed: int = 0,
    ):
        self.jitter = jitter
        self.vi_max_iters = vi_max_iters
        self.vi_tol = vi_tol
        self.pow_sobol_fit = int(pow_sobol_fit)
        self.n_restarts_fit = int(n_restarts_fit)
        self.sqp_config = sqp_config
        self.seed = int(seed)
        self.params = None
        self._X = None
        self._L = None
        self._alpha = None
        self._omega = None
        self.round_info: tuple = ()

    def fit(self, X, k, N, round_info=()):
        from hydro_bo.opt.solvers import multistart_sqp

        X = jnp.asarray(X, dtype=jnp.float64)
        k = jnp.asarray(k, dtype=jnp.float64).reshape(-1)
        N = jnp.asarray(N, dtype=jnp.float64).reshape(-1)
        self.round_info = tuple(round_info)
        d = int(X.shape[1])
        jitter = jnp.asarray(self.jitter, dtype=jnp.float64)

        # Decision vector: [log_amp, log_ls (d)] — no mean (zero-mean prior).
        lb = np.concatenate([[-5.0], np.full(d, -5.0)])
        ub = np.concatenate([[5.0], np.full(d, 5.0)])

        vi_max_iters = self.vi_max_iters
        vi_tol = self.vi_tol
        round_info_static = self.round_info

        def _nelbo_obj(x_vec, _p):
            params = {
                "log_amp": x_vec[0],
                "log_ls": jax.lax.dynamic_slice(x_vec, (1,), (d,)),
            }
            return _neg_elbo(
                params, X, k, N, jitter, round_info_static, vi_max_iters, vi_tol,
            ).reshape(())

        x_best, val_best, info = multistart_sqp(
            _nelbo_obj,
            (lb, ub),
            seed=self.seed + len(self.round_info),
            pow_sobol=self.pow_sobol_fit,
            n_restarts=self.n_restarts_fit,
            sqp_config=self.sqp_config,
        )

        self._release_buffers()
        self.params = {
            "log_amp": jnp.asarray(x_best[0], dtype=jnp.float64),
            "log_ls": jnp.asarray(x_best[1 : 1 + d], dtype=jnp.float64),
        }
        from hydro_bo.utils.logging_config import get_logger
        get_logger(__name__).debug(
            "binomial_gp_fit_complete",
            final_nelbo=val_best,
            n_starts=info["n_starts"],
            converged=info["converged"],
        )

        # Final VI inner loop at converged hyperparameters
        omega, _, _ = _vi_inner_loop(
            self.params, X, k, N, self.jitter, self.round_info,
            self.vi_max_iters, self.vi_tol,
        )

        # Cache prediction state
        kappa = k - N / 2.0
        y_tilde = kappa / omega
        noise = 1.0 / omega
        params_zero_mean = {**self.params, "mean": jnp.array(0.0, dtype=jnp.float64)}
        L, alpha = _factorise(
            params_zero_mean,
            X,
            y_tilde,
            noise,
            jnp.float64(self.jitter),
            self.round_info,
        )

        self._X = X
        self._L = L
        self._alpha = alpha
        self._omega = omega

    def predict(self, X_test):
        """Returns (mean, var) for q(f_*) at X_test — i.e. the latent posterior.
        Push through Φ at the call site for chance-constraint probabilities."""

        X_test = jnp.asarray(X_test, dtype=jnp.float64)
        if X_test.ndim == 1:
            X_test = X_test[None, :]

        return _predict_zero_mean(
            self.params,
            self._X,
            self._L,
            self._alpha,
            X_test,
            self.round_info,
        )

    def state(self) -> dict:
        """Snapshot of the fitted state in the dict shape consumed by
        `_predict` / `_feasibility_eval`. Mean is fixed at 0 (zero-mean
        prior under the PG augmentation)."""
        if self.params is None:
            raise RuntimeError("BinomialGP.state() called before fit()")
        return {
            "params": self.params,
            "X": self._X,
            "L": self._L,
            "alpha": self._alpha,
            "mean": jnp.asarray(0.0, dtype=jnp.float64),
        }

    def probability_feasibile(self, X_test, alpha):
        """P(σ(f_*) ≥ alpha) at each test point."""
        mu, var = self.predict(X_test)
        sigma = jnp.sqrt(var)
        threshold = jnp.log(alpha / (1.0 - alpha))
        z = (mu - threshold) / sigma
        return jax.scipy.stats.norm.cdf(z)
