"""Bayesian optimisers.

`BaseBayesopt` owns the loop, the initial-design phase, snap-to-grid for
categorical/integer dims, the round_info derivation, and the public API
(`run`, `suggest`, `observe`). Subclasses plug in:

  - `_fit_surrogates()`: how to (re)fit the surrogate(s) from `self.dataset`.
  - `_build_acquisition()`: which `AcquisitionFunction` to optimise next.
  - `_best_observed()`: how to score training points and pick the incumbent.

Implementations:

  - `MeanVarBayesopt`: heteroscedastic mean + log-variance GPs and noisy
    EI on g(x) = mu(x) - lam * sigma(x). The production path.
  - `ConstrainedBayesopt`: cEI with a Polya-Gamma augmented
    BinomialGP for feasibility.
"""

from abc import ABC, abstractmethod
from typing import Callable, Sequence, Tuple

import gc
import numpy as np
import jax.numpy as jnp

from hydro_bo.opt.acquisition import (
    AcquisitionFunction,
    ConstrainedExpectedImprovement,
    ExpectedImprovement,
    _ei_g_stats,
)
from hydro_bo.opt.dataset import Dataset
from hydro_bo.opt.solvers import (
    ConstrainedMixedIntNLP,
    MixedIntNLP,
    NLPBase,
    sobol_sample,
)
from hydro_bo.opt.surrogate import BinomialGP, HeteroscedasticGP
from hydro_bo.utils.logging_config import get_logger

logger = get_logger(__name__)


class BaseBayesopt(ABC):
    """BO with a stochastic objective f(x) -> sample array.

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
        sqp_config=None,
        pad_initial: int = 256,
        gp_lbfgs_max_iter: int = 100,
    ):

        self.f = f
        self.bounds = np.asarray(bounds, dtype=float)
        self.n_initial_points = n_initial_points
        self.iter_limit = iter_limit
        self.lam = lam
        self.n_restarts = n_restarts
        self.pow_sobol = pow_sobol
        self.seed = seed
        self.cat_vars = [(int(i), [float(v) for v in vals]) for i, vals in cat_vars]
        self.sqp_config = sqp_config
        self.pad_initial = int(pad_initial)
        self.gp_lbfgs_max_iter = int(gp_lbfgs_max_iter)

        self._X: list[np.ndarray] = []
        self._samples: list[np.ndarray] = []
        self.dataset: Dataset | None = None

    # ---------- subclass hooks ----------

    @abstractmethod
    def _fit_surrogates(self) -> None:
        """Refit surrogate(s) from `self.dataset`. Called at the start of
        every BO iteration — must update whatever state
        `_build_acquisition` depends on."""
        ...

    @abstractmethod
    def _build_acquisition(self) -> AcquisitionFunction:
        """Construct the acquisition object to be optimised next.
        Should pull GP state via `gp.state()` and pass the same
        round_info that was used to fit those GPs."""
        ...

    @abstractmethod
    def _build_solver(self, seed: int) -> NLPBase:
        """Construct the NLP solver used to maximise the acquisition.
        `seed` is the per-iteration seed (Sobol scramble)."""
        ...

    @abstractmethod
    def _best_observed(self) -> tuple[np.ndarray, float]:
        """Best observed point and its score, by whatever metric this
        flavour of BO cares about."""
        ...

    # ---------- shared infrastructure ----------

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

    def _build_round_info(self) -> tuple:
        """Derive the kernel round_info from self.cat_vars. Empty
        cat_vars → empty round_info, i.e. kernel stays continuous."""
        if not self.cat_vars:
            return ()
        info = sorted(
            ((int(idx), int(len(positions))) for idx, positions in self.cat_vars),
            key=lambda p: p[0],
        )
        return tuple(info)

    def observe(self, x: np.ndarray, samples) -> None:
        """Record an observation: x in original bounds, samples = 1D iterable.
        Non-finite entries (NaN / +/-inf) are preserved — they're how the
        objective signals failed solves to the constrained BO. Mean and
        variance are computed downstream over only the finite subset.
        Integer dims of x are snapped to their grid before storage."""
        x = self._snap_to_grid(x)
        s = np.asarray(samples, dtype=float).ravel()
        self._X.append(x)
        self._samples.append(s)

    def _evaluate_and_store(self, x: np.ndarray) -> np.ndarray:
        x = self._snap_to_grid(x)
        s = np.asarray(self.f(x), dtype=float).ravel()
        self._X.append(x)
        self._samples.append(s)
        return s

    def run(self) -> tuple[np.ndarray, float]:
        n_preloaded = len(self._X)
        n_to_sample = max(0, self.n_initial_points - n_preloaded)
        if n_to_sample > 0:
            logger.info(
                "bo_sobol_phase_start", n_to_sample=n_to_sample, n_preloaded=n_preloaded
            )
            X_init = sobol_sample(self.bounds, n_to_sample, seed=self.seed)
            for i, x in enumerate(X_init):
                s = self._evaluate_and_store(x)
                logger.info(
                    "bo_sobol_observation",
                    point=i + 1,
                    n_total=n_to_sample,
                    n_samples=int(len(s)),
                    n_feasible=int(np.sum(np.isfinite(s))),
                    mean=float(np.nanmean(s)) if np.any(np.isfinite(s)) else float("nan"),
                )
        else:
            logger.info("bo_sobol_phase_skipped", n_preloaded=n_preloaded)

        logger.info("bo_phase_start", iter_limit=self.iter_limit, lam=self.lam)
        try:
            import psutil

            _proc = psutil.Process()
        except Exception:
            _proc = None
        for i in range(self.iter_limit):
            self._fit_surrogates()
            x_next = self._suggest_next()
            s = self._evaluate_and_store(x_next)
            best_x, best_score = self._best_observed()
            rss_mb = (
                float(_proc.memory_info().rss) / (1024**2)
                if _proc is not None
                else float("nan")
            )
            n_feas = int(np.sum(np.isfinite(s)))
            logger.info(
                "bo_iteration",
                iteration=i + 1,
                iter_limit=self.iter_limit,
                n_samples=int(len(s)),
                n_feasible=n_feas,
                mean=float(np.nanmean(s)) if n_feas else float("nan"),
                var=float(np.nanvar(s, ddof=1)) if n_feas >= 2 else float("nan"),
                best_score=float(best_score),
                driver_rss_mb=rss_mb,
            )

        best_x, best_score = self._best_observed()
        logger.info(
            "bo_complete",
            best_score=float(best_score),
            n_total_evals=len(self._samples),
        )
        return best_x, best_score

    def suggest(self) -> np.ndarray:
        self._fit_surrogates()
        return self._suggest_next()

    def _suggest_next(self) -> np.ndarray:
        if self.dataset is None:
            self._fit_surrogates()
        logger.debug(
            "bo_acquisition_optimise", lam=self.lam, n_restarts=self.n_restarts
        )
        acq = self._build_acquisition()
        seed = self.seed + len(self._X)
        solver = self._build_solver(seed=seed)
        best_x_unit, best_val = solver.maximise(acq)

        # Predicted confidence-bound g(x) = mu - lam·sigma at the chosen
        # point, in the original (un-standardised) target space —
        # `_ei_g_stats` undoes the GP-target scaling internally.
        e_g, var_g = _ei_g_stats(
            jnp.asarray(best_x_unit, dtype=jnp.float64)[None, :],
            acq.gp_mu_state,
            acq.gp_log_var_state,
            acq.scaling,
            acq.round_info,
            acq.mu_kernel_kind,
            acq.lv_kernel_kind,
        )
        cb = float(e_g[0])
        cb_sigma = float(jnp.sqrt(jnp.clip(var_g[0], 0.0, None)))
        logger.info(
            "bo_acquisition_optimise_complete",
            ei_value=float(best_val),
            confidence_bound=cb,
            confidence_bound_sigma=cb_sigma,
            incumbent_g_best=float(acq._g_best),
        )
        x_orig = self.dataset.to_original(np.asarray(best_x_unit))

        del acq, solver
        # No `clear_jax_caches()` — observations are padded to `n_max`
        # so the JIT cache is shape-stable across BO iterations and
        # reusing it is the whole point.
        gc.collect()

        return x_orig


class MeanVarBayesopt(BaseBayesopt):
    """Heteroscedastic mean + log-variance GPs, noisy EI on
    g(x) = mu(x) - lam * sigma(x).
    """

    def __init__(
        self,
        *args,
        gp_mu_kernel: str = "rbf",
        gp_log_var_kernel: str = "rbf",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.gp_mu_kernel = str(gp_mu_kernel)
        self.gp_log_var_kernel = str(gp_log_var_kernel)
        self.gp_mu = HeteroscedasticGP(
            pad_initial=self.pad_initial,
            lbfgs_max_iter=self.gp_lbfgs_max_iter,
            seed=self.seed,
            kernel_kind=self.gp_mu_kernel,
        )
        self.gp_log_var = HeteroscedasticGP(
            pad_initial=self.pad_initial,
            lbfgs_max_iter=self.gp_lbfgs_max_iter,
            seed=self.seed + 1,
            kernel_kind=self.gp_log_var_kernel,
        )

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

    def _fit_surrogates(self) -> None:
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
            noise_lv = ds.noise_log_sigma2[m_lv] / (ds._sigma_lv**2)
            self.gp_log_var.fit(X_lv, y_lv, noise_lv, round_info=round_info)
            log_var_fit = True
            logger.info("bo_log_var_gp_fit", n_points=int(m_lv.sum()))
        else:
            logger.warning(
                "bo_log_var_gp_skipped",
                n_valid=int(m_lv.sum()),
                reason="need >= 2 points with n_samples >= 2",
            )

        # 2) Mean GP on rows with n >= 1, with noise = predicted-pop-var / n
        # (fall back to empirical s^2 if log-var GP couldn't be fit yet).
        m_mu = ds.mask_mu
        if int(m_mu.sum()) < 2:
            raise RuntimeError(
                f"Mean GP needs >= 2 valid mean observations, got {int(m_mu.sum())}"
            )
        X_mu = X_unit[m_mu]
        y_mu = ds.mu_scaled[m_mu]
        n_mu = ds.k[m_mu].astype(float)

        if log_var_fit:
            log_v_pred_s, log_v_pred_v = self.gp_log_var.predict(
                jnp.asarray(X_mu, dtype=jnp.float64)
            )
            log_v_pred = np.asarray(log_v_pred_s) * ds._sigma_lv + ds._mu_lv
            log_v_pred_var = np.asarray(log_v_pred_v) * ds._sigma_lv**2
            pop_var = np.exp(log_v_pred + 1 / 2 * log_v_pred_var)
        else:
            sv = ds.sigma2[m_mu]
            mean_finite = float(np.nanmean(sv)) if np.isfinite(sv).any() else 1.0
            pop_var = np.where(np.isfinite(sv), sv, mean_finite)

        noise_mu = pop_var / np.maximum(n_mu, 1.0)
        noise_mu_scaled = noise_mu / (ds._sigma_y**2)
        self.gp_mu.fit(X_mu, y_mu, noise_mu_scaled, round_info=round_info)
        logger.info(
            "bo_mean_gp_fit", n_points=int(m_mu.sum()), used_log_var_gp=log_var_fit
        )

    def _build_acquisition(self) -> ExpectedImprovement:
        acq = ExpectedImprovement(
            self.gp_mu,
            self.gp_log_var,
            self.lam,
            self.dataset,
            round_info=self.gp_mu.round_info,
        )
        logger.debug("bo_noisy_ei_incumbent", g_best=float(acq._g_best))
        return acq

    def _build_solver(self, seed: int) -> MixedIntNLP:
        return MixedIntNLP(
            cat_vars=self.cat_vars,
            seed=seed,
            pow_sobol=self.pow_sobol,
            n_restarts=self.n_restarts,
            sqp_config=self.sqp_config,
        )


class ConstrainedBayesopt(MeanVarBayesopt):
    """MeanVarBayesopt + a Polya-Gamma augmented BinomialGP feasibility
    surrogate and a chance-bound constraint enforced inside the SQP.

    Inherits the mean / log-variance GP fit from `MeanVarBayesopt`
    unchanged; adds `gp_bin` fit on (k, N) derived from the Dataset.
    Acquisition is `ConstrainedExpectedImprovement` (EI on the mu/log-var
    GPs with a latent-space chance-bound constraint
    `f(x) + z_sc·σ(x) − log_p_targ ≥ 0`); solver is
    `ConstrainedMixedIntNLP` (which adds the SQP constraint and an L1
    hinge in the Sobol screen).

    `_best_observed` is overridden to filter for empirical feasibility
    (`k_i / N_i ≥ p_targ`) before scoring.
    """

    def __init__(
        self,
        *args,
        p_targ: float = 0.9,
        z_sc: float = 1.6449,  # Φ⁻¹(0.95) — pass directly as z-score, not as a probability.
        l1_penalty: float = 1.0,
        gp_bin_kernel: str = "matern12",
        gp_bin_warp_dims: tuple = (),
        gp_bin_prior_warp_scale: float = 0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.p_targ = float(p_targ)
        self.z_sc = float(z_sc)
        self.l1_penalty = float(l1_penalty)
        self.gp_bin_kernel = str(gp_bin_kernel)
        self.gp_bin_warp_dims = tuple(int(i) for i in gp_bin_warp_dims)
        self.gp_bin_prior_warp_scale = float(gp_bin_prior_warp_scale)
        self.gp_bin = BinomialGP(
            pad_initial=self.pad_initial,
            lbfgs_max_iter=self.gp_lbfgs_max_iter,
            seed=self.seed + 2,
            kernel_kind=self.gp_bin_kernel,
            warp_dims=self.gp_bin_warp_dims,
            prior_warp_scale=self.gp_bin_prior_warp_scale,
        )

    def _fit_surrogates(self) -> None:
        # Mean + log-variance GPs (and self.dataset) — full MeanVar machinery.
        super()._fit_surrogates()
        # Add the binomial feasibility surrogate.
        ds = self.dataset
        m_bin = ds.mask_bin
        n_bin = int(m_bin.sum())
        if n_bin < 2:
            logger.warning(
                "bo_bin_gp_skipped",
                n_valid=n_bin,
                reason="need >= 2 observed points",
            )
            return
        X_bin = ds.X_scaled[m_bin]
        k_bin = ds.k[m_bin].astype(float)
        N_bin = ds.N[m_bin].astype(float)
        round_info = self._build_round_info()
        self.gp_bin.fit(X_bin, k_bin, N_bin, round_info=round_info)
        logger.info(
            "bo_bin_gp_fit",
            n_points=n_bin,
            n_feasible_total=int(np.sum(k_bin)),
            n_trials_total=int(np.sum(N_bin)),
        )

    def _build_acquisition(self) -> ConstrainedExpectedImprovement:
        return ConstrainedExpectedImprovement(
            gp_mu=self.gp_mu,
            gp_log_var=self.gp_log_var,
            gp_bin=self.gp_bin,
            lam=self.lam,
            p_targ=self.p_targ,
            z_sc=self.z_sc,
            dataset=self.dataset,
            round_info=self.gp_mu.round_info,
        )

    def _build_solver(self, seed: int) -> ConstrainedMixedIntNLP:
        return ConstrainedMixedIntNLP(
            cat_vars=self.cat_vars,
            l1_penalty=self.l1_penalty,
            seed=seed,
            pow_sobol=self.pow_sobol,
            n_restarts=self.n_restarts,
            sqp_config=self.sqp_config,
        )

    def _best_observed(self) -> tuple[np.ndarray, float]:
        """Best (x, mu - lam·sd) over training points whose empirical
        feasibility rate `k/N` meets `p_targ`. Falls back to the
        unconstrained best (with a warning) when no observation is yet
        empirically feasible."""
        ds = Dataset(np.stack(self._X), self._samples, self.bounds)
        sd = np.where(np.isfinite(ds.sigma2), np.sqrt(np.maximum(ds.sigma2, 0.0)), 0.0)
        scores = ds.mu - self.lam * sd
        N_safe = np.maximum(ds.N, 1)
        feasible = (ds.k / N_safe) >= self.p_targ
        valid = np.isfinite(scores) & feasible
        if not valid.any():
            logger.warning(
                "bo_no_feasible_incumbent",
                p_targ=self.p_targ,
                fallback="unconstrained best",
            )
            valid = np.isfinite(scores)
            if not valid.any():
                return self._X[0], float("nan")
        idx_in_valid = int(np.argmax(scores[valid]))
        idx = int(np.flatnonzero(valid)[idx_in_valid])
        return self._X[idx], float(scores[idx])
