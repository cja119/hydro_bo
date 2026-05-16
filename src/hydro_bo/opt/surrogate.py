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


_KERNEL_KINDS = ("rbf", "matern12", "matern52")


@partial(jax.jit, static_argnames=("round_info", "kernel_kind"))
def _kernel(log_amp, log_ls, X1, X2, round_info, kernel_kind):
    """ARD kernel dispatcher with optional projection of integer dims
    onto a discrete unit-cube grid. `kernel_kind` is a static Python
    string. round_info is a static Python tuple — empty skips projection.

    Kernels (all with ARD per-dim length scales):
      - `"rbf"`      : squared-exponential, C^∞. Smoothest; for smooth
                       functions, but length-scale fits can run away.
      - `"matern12"` : exponential / Laplacian, C^0. Sharpest decay;
                       represents discontinuities but penalises smoothness.
      - `"matern52"` : Matérn ν=5/2, C^2. The BO-literature default for
                       continuous inputs (Snoek et al. 2012, Gardner et al.
                       2014). Smoother than matern12 — robust middle
                       ground; less prone to length-scale runaway when
                       combined with hyperpriors.
    """
    if round_info:
        X1 = _project_integer_dims(X1, round_info)
        X2 = _project_integer_dims(X2, round_info)
    amp2 = jnp.exp(2.0 * log_amp)
    ls = jnp.exp(log_ls)
    diff = (X1[:, None, :] - X2[None, :, :]) / ls

    if kernel_kind == "rbf":
        d2 = jnp.sum(diff * diff, axis=-1)
        return amp2 * jnp.exp(-0.5 * d2)
    if kernel_kind == "matern12":
        eps = 1e-12
        d1 = jnp.sum(jnp.abs(diff), axis=-1) + eps
        return amp2 * jnp.exp(-d1)
    if kernel_kind == "matern52":
        # d = sqrt(sum (Δx/ls)^2). Use the L2 distance per Rasmussen &
        # Williams (eq. 4.17). Adding a tiny jitter under the sqrt keeps
        # the gradient finite at d=0 (the formula is smooth in d but its
        # autodiff through sqrt blows up at zero distance).
        d2 = jnp.sum(diff * diff, axis=-1)
        d = jnp.sqrt(d2 + 1e-12)
        s5 = jnp.sqrt(5.0)
        return amp2 * (1.0 + s5 * d + (5.0 / 3.0) * d2) * jnp.exp(-s5 * d)
    raise ValueError(f"unknown kernel_kind: {kernel_kind!r}; expected one of {_KERNEL_KINDS}")


def _build_K_masked(log_amp, log_ls, X, mask, noise, jitter, round_info, kernel_kind):
    """Build the (n_max, n_max) kernel matrix with padded rows/cols
    replaced by an identity block.

    For real rows i, j (mask=1): K_ij = kernel(x_i, x_j) (+ noise+jitter on diag).
    For padded rows i (mask=0): row/col zeroed except K_ii = 1 (jitter on diag).

    The Cholesky of this block-diagonal-ish matrix is well-conditioned;
    log|K| picks up `log(1) = 0` for each padded row (constant offset
    that doesn't affect the optimum); padded entries of `cho_solve(L, y_c)`
    are zero when y_c is masked, so prediction-time formulas need only
    mask the variance summation over training rows."""
    n_max = X.shape[0]
    K = _kernel(log_amp, log_ls, X, X, round_info, kernel_kind)
    M2 = mask[:, None] * mask[None, :]
    K = K * M2  # zero cross-terms involving any padded row/col
    diag_real = mask * (noise + jitter)  # noise+jitter on real diagonal
    diag_padded = 1.0 - mask  # 1.0 on padded diagonal
    return K + jnp.diag(diag_real + diag_padded)


def _neg_log_hyper_prior(log_amp, log_ls, alpha_ls, beta_ls, alpha_amp, beta_amp):
    """Negative log prior on (amp, ls) for type-II MAP estimation.

    Gamma priors on the natural-space (amp, ls) parameters, with the
    change-of-variable jacobian for the log-space optimisation. For ls ~
    Gamma(α, β) the density of log_ls = log(ls) is
        p(log_ls) ∝ exp(α · log_ls − β · exp(log_ls))
    so the negative log prior contribution (up to constant) is
        α · log_ls − β · exp(log_ls)        ... negated for minimisation
    Summed over all ARD length-scale dims, plus the amp term.

    Defaults follow BoTorch's `SingleTaskGP`: Gamma(3, 6) on length
    scale (mean=0.5, mode=0.33, soft upper bound ~1.5) and Gamma(2, 0.5)
    on amp (mean=4, mode=2). The length-scale prior is the key one for
    avoiding the runaway-to-infinity pathology under type-II MLE; the
    amp prior is light and mostly there for numerical stability.

    Set alpha=beta=0 on a parameter to disable its prior entirely
    (recovers the unpenalised MLE/ELBO).
    """
    # ls prior: sum over ARD dims of -α·log_ls + β·exp(log_ls)
    neg_log_ls_prior = jnp.where(
        (alpha_ls > 0) | (beta_ls > 0),
        jnp.sum(-alpha_ls * log_ls + beta_ls * jnp.exp(log_ls)),
        0.0,
    )
    # amp prior: -α_amp·log_amp + β_amp·exp(log_amp)
    neg_log_amp_prior = jnp.where(
        (alpha_amp > 0) | (beta_amp > 0),
        -alpha_amp * log_amp + beta_amp * jnp.exp(log_amp),
        0.0,
    )
    return neg_log_ls_prior + neg_log_amp_prior


@partial(jax.jit, static_argnames=("round_info", "kernel_kind"))
def _neg_mll(params, X, y, noise, mask, jitter,
             alpha_ls, beta_ls, alpha_amp, beta_amp,
             round_info, kernel_kind):
    """Masked NMLL + neg-log-prior on hyperparameters (type-II MAP).
    Observations on `mask == 1` rows count, padded rows contribute a
    constant offset that drops out of optimisation. The hyperprior
    contribution is the standard MAP penalty — see `_neg_log_hyper_prior`.
    """
    K = _build_K_masked(
        params["log_amp"],
        params["log_ls"],
        X,
        mask,
        noise,
        jitter,
        round_info,
        kernel_kind,
    )
    y_c = (y - params["mean"]) * mask
    L = jnp.linalg.cholesky(K)
    alpha = jax.scipy.linalg.cho_solve((L, True), y_c)
    log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    n_real = jnp.sum(mask)
    nmll = 0.5 * (jnp.dot(y_c, alpha) + log_det + n_real * jnp.log(2.0 * jnp.pi))
    prior = _neg_log_hyper_prior(
        params["log_amp"], params["log_ls"], alpha_ls, beta_ls, alpha_amp, beta_amp,
    )
    return nmll + prior


@partial(jax.jit, static_argnames=("round_info", "kernel_kind"))
def _factorise(params, X, y, noise, mask, jitter, round_info, kernel_kind):
    K = _build_K_masked(
        params["log_amp"],
        params["log_ls"],
        X,
        mask,
        noise,
        jitter,
        round_info,
        kernel_kind,
    )
    y_c = (y - params["mean"]) * mask
    L = jnp.linalg.cholesky(K)
    alpha = jax.scipy.linalg.cho_solve((L, True), y_c)
    return L, alpha


@partial(jax.jit, static_argnames=("round_info", "kernel_kind"))
def _predict(params, X_train, L, alpha, mean, mask, X_test, round_info, kernel_kind):
    """Padded predict. `mask` zeros padded training rows out of the
    variance summation; padded contributions to the mean are already
    zero because `alpha[padded] == 0`."""
    K_s = _kernel(
        params["log_amp"], params["log_ls"], X_test, X_train, round_info, kernel_kind,
    )
    amp2 = jnp.exp(2.0 * params["log_amp"])
    mu = mean + K_s @ alpha
    v = jax.scipy.linalg.cho_solve((L, True), K_s.T)
    var = amp2 - jnp.sum(K_s * v.T * mask[None, :], axis=1)
    var = jnp.clip(var, 1e-12, None)
    return mu, var


@partial(jax.jit, static_argnames=("round_info", "kernel_kind"))
def _predict_zero_mean(params, X_train, L, alpha, mask, X_test, round_info, kernel_kind):
    params_zero_mean = {**params, "mean": jnp.array(0.0, dtype=jnp.float64)}
    return _predict(
        params_zero_mean,
        X_train,
        L,
        alpha,
        params_zero_mean["mean"],
        mask,
        X_test,
        round_info,
        kernel_kind,
    )


@partial(jax.jit, static_argnames=("round_info", "vi_max_iters", "vi_tol", "kernel_kind"))
def _vi_inner_loop(
    params, X, k, N, mask, jitter, round_info, vi_max_iters, vi_tol, kernel_kind,
):
    """PG-VI alternating updates with per-row mask. Padded rows have
    `mask=0`, `N=0`, `k=0`; we hold their `ω = 1` so synthetic noise
    `1/ω` stays finite and the masked kernel block stays well-conditioned."""
    n_max = X.shape[0]

    # Masked K: real-real block has the kernel + jitter on diag; padded
    # block is identity. Cross-terms zeroed.
    K = _build_K_masked(
        params["log_amp"], params["log_ls"], X, mask,
        jnp.zeros(n_max, dtype=X.dtype), jitter, round_info, kernel_kind,
    )
    kappa = (k - N / 2.0) * mask  # zero on padded

    def step(omega):
        noise = 1.0 / omega
        L = jnp.linalg.cholesky(K + jnp.diag(noise))
        alpha = jax.scipy.linalg.cho_solve((L, True), kappa / omega)
        m = K @ alpha
        v = jax.scipy.linalg.cho_solve((L, True), K)
        S_diag = jnp.clip(jnp.diag(K) - jnp.sum(K * v.T, axis=1), 1e-12, None)
        c = jnp.maximum(jnp.sqrt(m**2 + S_diag), 1e-6)
        omega_real = N / (2.0 * c) * jnp.tanh(c / 2.0)
        omega_new = jnp.where(mask > 0.5, omega_real, jnp.ones_like(omega))
        return omega_new, m, S_diag

    omega0 = jnp.where(mask > 0.5, N / 4.0, jnp.ones_like(N))
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


@partial(jax.jit, static_argnames=("round_info", "vi_max_iters", "vi_tol", "kernel_kind"))
def _neg_elbo(
    params, X, k, N, mask, jitter,
    alpha_ls, beta_ls, alpha_amp, beta_amp,
    round_info, vi_max_iters, vi_tol, kernel_kind,
):
    """Negative ELBO for PG-augmented Binomial GP + neg-log-prior on
    hyperparameters (type-II MAP). Padded rows (mask=0) contribute
    constants that don't depend on the hyperparameters — they're
    explicitly zeroed in every per-row sum so the optimum matches a fit
    on the real subset only. The hyperprior contribution is the standard
    MAP penalty — see `_neg_log_hyper_prior`.
    """
    n_real = jnp.sum(mask)

    # 1. Inner VI loop to converged (ω, m, S_diag) — masked.
    omega, m, S_diag = _vi_inner_loop(
        params, X, k, N, mask, jitter, round_info, vi_max_iters, vi_tol, kernel_kind,
    )

    # 2. Masked kernel — same trick as the heteroscedastic path.
    n_max = X.shape[0]
    K = _build_K_masked(
        params["log_amp"], params["log_ls"], X, mask,
        jnp.zeros(n_max, dtype=X.dtype), jitter, round_info, kernel_kind,
    )

    # 3. Convenience.
    kappa = (k - N / 2.0) * mask
    c = jnp.sqrt(m**2 + S_diag)
    noise = 1.0 / omega

    L_kn = jnp.linalg.cholesky(K + jnp.diag(noise))
    L_K = jnp.linalg.cholesky(K)

    # Expected log-lik (kappa, m, S_diag are already zero/masked on padded).
    expected_loglik = jnp.dot(kappa, m) - 0.5 * jnp.sum(mask * omega * (m**2 + S_diag))

    # PG KL — masked sum.
    pg_kl = jnp.sum(mask * (N * jnp.log(jnp.cosh(c / 2.0)) - 0.5 * c**2 * omega))

    # Gaussian KL — every per-row term explicitly masked. log-det
    # contributions on padded rows are constants (zero from L_K, log√2
    # from L_kn) which we drop via mask.
    trace_K_inv_S = n_real - jnp.sum(mask * omega * S_diag)
    K_inv_m = jax.scipy.linalg.cho_solve((L_K, True), m)
    quad_form = jnp.dot(m, K_inv_m)  # m is already 0 on padded
    log_det_K = 2.0 * jnp.sum(jnp.log(jnp.diag(L_K)) * mask)
    log_det_S = (
        log_det_K
        - 2.0 * jnp.sum(jnp.log(jnp.diag(L_kn)) * mask)
        - jnp.sum(jnp.log(omega) * mask)
    )
    gaussian_kl = 0.5 * (trace_K_inv_S + quad_form - n_real + log_det_K - log_det_S)

    elbo = expected_loglik - gaussian_kl - pg_kl
    prior = _neg_log_hyper_prior(
        params["log_amp"], params["log_ls"], alpha_ls, beta_ls, alpha_amp, beta_amp,
    )
    return -elbo + prior


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
        for attr in ("_X", "_L", "_alpha", "_mean", "_mask", "_omega"):
            buf = getattr(self, attr, None)
            if buf is not None and hasattr(buf, "delete"):
                try:
                    buf.delete()
                except Exception:
                    pass
            setattr(self, attr, None)

    def _ensure_pad(self, n_real: int) -> None:
        """Power-of-2 expanding pad. If `n_real` exceeds the current
        pad size, double until it fits and clear the JIT cache so the
        next fit retraces at the new shape. Across a full BO run this
        is O(log_2(n_final / n_initial)) recompiles instead of O(n)."""
        if n_real <= self.n_max:
            return
        from hydro_bo.opt.solvers import clear_jax_caches
        from hydro_bo.utils.logging_config import get_logger

        new_n_max = self.n_max
        while new_n_max < n_real:
            new_n_max *= 2
        get_logger(__name__).info(
            "gp_pad_grow",
            gp=type(self).__name__,
            old_n_max=self.n_max,
            new_n_max=new_n_max,
            n_real=n_real,
        )
        self.n_max = new_n_max
        # Drop stale compiled artifacts at the old shape — the next
        # fit / predict will retrace at `new_n_max`.
        clear_jax_caches()


class HeteroscedasticGP(BaseGP):
    """Exact GP with constant mean, fixed per-point noise; kernel
    selectable via `kernel_kind` ("rbf" or "matern12", ARD in either
    case).

    Observations are padded to `n_max` rows up front; the JIT cache is
    keyed on this constant shape, so the BO loop pays one trace and
    reuses it across iterations. Hyperparameters are fit via jaxopt
    L-BFGS warm-started from the previous iteration's converged params.
    """

    def __init__(
        self,
        pad_initial: int,
        jitter: float = 1e-6,
        lbfgs_max_iter: int = 100,
        seed: int = 0,
        kernel_kind: str = "rbf",
        # Gamma(α, β) hyperpriors on ls and amp for type-II MAP. Defaults
        # mirror BoTorch's SingleTaskGP — Gamma(3, 6) on ls (mean=0.5,
        # mode=0.33) softly bounds length scales below ~1.5. Set both
        # alpha and beta to 0 to disable the prior (recovers MLE).
        prior_ls_alpha: float = 3.0,
        prior_ls_beta: float = 6.0,
        prior_amp_alpha: float = 2.0,
        prior_amp_beta: float = 0.5,
    ):
        if kernel_kind not in _KERNEL_KINDS:
            raise ValueError(
                f"unknown kernel_kind: {kernel_kind!r}; expected one of {_KERNEL_KINDS}"
            )
        self.pad_initial = int(pad_initial)
        self.n_max = int(pad_initial)  # current pad; doubles on overflow
        self.jitter = jitter
        self.lbfgs_max_iter = int(lbfgs_max_iter)
        self.seed = int(seed)
        self.kernel_kind = str(kernel_kind)
        self.prior_ls_alpha = float(prior_ls_alpha)
        self.prior_ls_beta = float(prior_ls_beta)
        self.prior_amp_alpha = float(prior_amp_alpha)
        self.prior_amp_beta = float(prior_amp_beta)
        self.params = None
        self._X = None
        self._L = None
        self._alpha = None
        self._mean = None
        self._mask = None
        self.round_info: tuple = ()

    def state(self) -> dict:
        """Snapshot of the fitted state needed by the acquisition.
        Includes the mask so predict-time kernels operate on the
        padded shape consistently."""
        return {
            "params": self.params,
            "X": self._X,
            "L": self._L,
            "alpha": self._alpha,
            "mean": self._mean,
            "mask": self._mask,
        }

    def fit(self, X, y, noise, round_info: tuple = ()):
        from jaxopt import LBFGS
        from hydro_bo.utils.logging_config import get_logger

        logger = get_logger(__name__)

        X_real = jnp.asarray(X, dtype=jnp.float64)
        y_real = jnp.asarray(y, dtype=jnp.float64).reshape(-1)
        noise_real = jnp.asarray(noise, dtype=jnp.float64).reshape(-1)
        n_real, d = int(X_real.shape[0]), int(X_real.shape[1])
        round_info = tuple(round_info)
        self.round_info = round_info
        jitter = jnp.asarray(self.jitter, dtype=jnp.float64)

        # Grow the pad if needed (power-of-2 doubling, clears JIT cache).
        self._ensure_pad(n_real)

        # Pad to (n_max, d) — padded rows carry zeros for X/y/noise; the
        # mask flags which rows are real. Masked kernel handles the rest.
        pad_n = self.n_max - n_real
        X_padded = jnp.concatenate([X_real, jnp.zeros((pad_n, d), dtype=jnp.float64)], axis=0)
        y_padded = jnp.concatenate([y_real, jnp.zeros(pad_n, dtype=jnp.float64)], axis=0)
        noise_padded = jnp.concatenate([noise_real, jnp.zeros(pad_n, dtype=jnp.float64)], axis=0)
        mask = jnp.concatenate([jnp.ones(n_real), jnp.zeros(pad_n)]).astype(jnp.float64)

        logger.debug(
            "gp_fit_start", n_real=n_real, n_max=self.n_max, n_dims=d,
            warm_start=self.params is not None,
        )

        # Warm start from previous converged params if available; cold
        # start otherwise.
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
                "mean": jnp.array(float(np.mean(np.asarray(y_real))), dtype=jnp.float64),
            }

        alpha_ls = jnp.asarray(self.prior_ls_alpha, dtype=jnp.float64)
        beta_ls = jnp.asarray(self.prior_ls_beta, dtype=jnp.float64)
        alpha_amp = jnp.asarray(self.prior_amp_alpha, dtype=jnp.float64)
        beta_amp = jnp.asarray(self.prior_amp_beta, dtype=jnp.float64)

        nmll = partial(_neg_mll, round_info=round_info, kernel_kind=self.kernel_kind)
        solver = LBFGS(fun=nmll, maxiter=self.lbfgs_max_iter)
        result = solver.run(
            init, X_padded, y_padded, noise_padded, mask, jitter,
            alpha_ls, beta_ls, alpha_amp, beta_amp,
        )

        self._release_buffers()
        self.params = {k: jnp.asarray(v) for k, v in result.params.items()}
        L, alpha = _factorise(
            self.params, X_padded, y_padded, noise_padded, mask, jitter, round_info,
            self.kernel_kind,
        )
        self._X = X_padded
        self._L = L
        self._alpha = alpha
        self._mean = self.params["mean"]
        self._mask = mask

        logger.debug(
            "gp_fit_complete",
            final_nmll=float(result.state.value),
            n_lbfgs_iters=int(result.state.iter_num),
        )

    def predict(self, X_test):
        """Returns (mean, var) at X_test in the GP's target space.
        The padded mask is threaded through `_predict` so the variance
        sum correctly excludes padded training rows."""
        X_test = jnp.asarray(X_test, dtype=jnp.float64)
        if X_test.ndim == 1:
            X_test = X_test[None, :]
        return _predict(
            self.params,
            self._X,
            self._L,
            self._alpha,
            self._mean,
            self._mask,
            X_test,
            self.round_info,
            self.kernel_kind,
        )


class BinomialGP(BaseGP):
    """Polya-Gamma augmented GP for binary / count feasibility data;
    kernel selectable via `kernel_kind` ("rbf" or "matern12"). Matérn-1/2
    is preferred for feasibility surrogates because it absorbs sharp
    feasibility boundaries (k=0 cluster next to k=N cluster) without
    forcing extreme length scales — which is what causes the σ-blowup
    that breaks chance-constrained acquisition optimisation under RBF.

    Padded to `n_max` rows to share the JIT cache across BO iterations;
    hyperparameters fit by jaxopt L-BFGS warm-started from the previous
    iteration's converged params. Inner PG-VI is masked so padded rows
    contribute no gradient."""

    def __init__(
        self,
        pad_initial: int,
        jitter: float = 1e-6,
        vi_max_iters: int = 30,
        vi_tol: float = 1e-4,
        lbfgs_max_iter: int = 100,
        seed: int = 0,
        kernel_kind: str = "matern12",
        # Gamma(α, β) hyperpriors — same defaults as HeteroscedasticGP.
        # The length-scale prior is what stops the runaway-to-infinity
        # pathology that collapses posterior σ across the cube and
        # breaks chance-constrained acquisition optimisation. See
        # `_neg_log_hyper_prior` for details.
        prior_ls_alpha: float = 3.0,
        prior_ls_beta: float = 6.0,
        prior_amp_alpha: float = 2.0,
        prior_amp_beta: float = 0.5,
    ):
        if kernel_kind not in _KERNEL_KINDS:
            raise ValueError(
                f"unknown kernel_kind: {kernel_kind!r}; expected one of {_KERNEL_KINDS}"
            )
        self.pad_initial = int(pad_initial)
        self.n_max = int(pad_initial)  # current pad; doubles on overflow
        self.jitter = jitter
        self.vi_max_iters = int(vi_max_iters)
        self.vi_tol = float(vi_tol)
        self.lbfgs_max_iter = int(lbfgs_max_iter)
        self.seed = int(seed)
        self.kernel_kind = str(kernel_kind)
        self.prior_ls_alpha = float(prior_ls_alpha)
        self.prior_ls_beta = float(prior_ls_beta)
        self.prior_amp_alpha = float(prior_amp_alpha)
        self.prior_amp_beta = float(prior_amp_beta)
        self.params = None
        self._X = None
        self._L = None
        self._alpha = None
        self._omega = None
        self._mask = None
        self.round_info: tuple = ()

    def fit(self, X, k, N, round_info=()):
        from jaxopt import LBFGS

        X_real = jnp.asarray(X, dtype=jnp.float64)
        k_real = jnp.asarray(k, dtype=jnp.float64).reshape(-1)
        N_real = jnp.asarray(N, dtype=jnp.float64).reshape(-1)
        n_real, d = int(X_real.shape[0]), int(X_real.shape[1])
        self.round_info = tuple(round_info)
        jitter = jnp.asarray(self.jitter, dtype=jnp.float64)

        self._ensure_pad(n_real)

        pad_n = self.n_max - n_real
        X_padded = jnp.concatenate([X_real, jnp.zeros((pad_n, d), dtype=jnp.float64)], axis=0)
        k_padded = jnp.concatenate([k_real, jnp.zeros(pad_n, dtype=jnp.float64)], axis=0)
        N_padded = jnp.concatenate([N_real, jnp.zeros(pad_n, dtype=jnp.float64)], axis=0)
        mask = jnp.concatenate([jnp.ones(n_real), jnp.zeros(pad_n)]).astype(jnp.float64)

        # Warm start if available; cold start otherwise.
        if self.params is not None and self.params["log_ls"].shape[0] == d:
            init = {
                "log_amp": self.params["log_amp"],
                "log_ls": self.params["log_ls"],
            }
        else:
            init = {
                "log_amp": jnp.array(0.0, dtype=jnp.float64),
                "log_ls": jnp.zeros(d, dtype=jnp.float64),
            }

        alpha_ls = jnp.asarray(self.prior_ls_alpha, dtype=jnp.float64)
        beta_ls = jnp.asarray(self.prior_ls_beta, dtype=jnp.float64)
        alpha_amp = jnp.asarray(self.prior_amp_alpha, dtype=jnp.float64)
        beta_amp = jnp.asarray(self.prior_amp_beta, dtype=jnp.float64)

        nelbo = partial(
            _neg_elbo,
            round_info=self.round_info,
            vi_max_iters=self.vi_max_iters,
            vi_tol=self.vi_tol,
            kernel_kind=self.kernel_kind,
        )
        solver = LBFGS(fun=nelbo, maxiter=self.lbfgs_max_iter)
        result = solver.run(
            init, X_padded, k_padded, N_padded, mask, jitter,
            alpha_ls, beta_ls, alpha_amp, beta_amp,
        )

        self._release_buffers()
        self.params = {key: jnp.asarray(v) for key, v in result.params.items()}

        # Final VI inner loop at converged hyperparameters → cached ω.
        omega, _, _ = _vi_inner_loop(
            self.params, X_padded, k_padded, N_padded, mask,
            self.jitter, self.round_info,
            self.vi_max_iters, self.vi_tol, self.kernel_kind,
        )

        # Cache prediction state via the masked factorise. Synthetic
        # targets/noise on padded rows are arbitrary (they're masked
        # out in `_factorise`'s y_c multiplication).
        kappa = (k_padded - N_padded / 2.0) * mask
        y_tilde = kappa / omega
        noise = 1.0 / omega
        params_zero_mean = {**self.params, "mean": jnp.array(0.0, dtype=jnp.float64)}
        L, alpha = _factorise(
            params_zero_mean, X_padded, y_tilde, noise, mask,
            jnp.float64(self.jitter), self.round_info, self.kernel_kind,
        )

        self._X = X_padded
        self._L = L
        self._alpha = alpha
        self._omega = omega
        self._mask = mask

        from hydro_bo.utils.logging_config import get_logger
        get_logger(__name__).debug(
            "binomial_gp_fit_complete",
            final_nelbo=float(result.state.value),
            n_lbfgs_iters=int(result.state.iter_num),
        )

    def predict(self, X_test):
        """Returns (mean, var) for q(f_*) at X_test — the latent posterior.
        Push through Φ at the call site for chance-constraint probabilities."""
        X_test = jnp.asarray(X_test, dtype=jnp.float64)
        if X_test.ndim == 1:
            X_test = X_test[None, :]
        return _predict_zero_mean(
            self.params,
            self._X,
            self._L,
            self._alpha,
            self._mask,
            X_test,
            self.round_info,
            self.kernel_kind,
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
            "mask": self._mask,
        }

    def probability_feasibile(self, X_test, alpha):
        """P(σ(f_*) ≥ alpha) at each test point."""
        mu, var = self.predict(X_test)
        sigma = jnp.sqrt(var)
        threshold = jnp.log(alpha / (1.0 - alpha))
        z = (mu - threshold) / sigma
        return jax.scipy.stats.norm.cdf(z)
