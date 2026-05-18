"""Dataset container for stochastic-objective Bayesian optimisation.

Pure-numpy. Holds raw inputs, per-input sample arrays, and derived
quantities used by the mean / log-variance GPs (sample mean, sample
variance, scaling parameters, validity masks).
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Dataset:
    _X: np.ndarray
    _samples: list
    bounds: np.ndarray

    def __post_init__(self):

        b = np.asarray(self.bounds, dtype=float)
        self._lo = b[:, 0]
        self._span = b[:, 1] - b[:, 0]
        samples = [np.asarray(s, dtype=float).ravel() for s in self._samples]
        self._samples = samples
        self.N = np.array([len(s) for s in samples], dtype=int)
        finite_subsets = [s[np.isfinite(s)] for s in samples]
        self.k = np.array([len(fs) for fs in finite_subsets], dtype=int)

        with np.errstate(invalid="ignore"):
            self.mu = np.array(
                [
                    float(np.mean(fs)) if len(fs) >= 1 else np.nan
                    for fs in finite_subsets
                ]
            )
            self.sigma2 = np.array(
                [
                    float(np.var(fs, ddof=1)) if len(fs) >= 2 else np.nan
                    for fs in finite_subsets
                ]
            )

        # Numerically stable log-variance, debiased under the asymptotic
        # chi-square approximation. log s² ~ N(log σ² − 1/(k−1), 2/(k−1))
        # for (k-1)·s²/σ² ~ χ²(k-1). Subtract the −1/(k-1) mean so the
        # target is an unbiased estimator of log σ². Pairs with
        # `noise_log_sigma2 = 2/(k-1)`, the matching asymptotic variance.
        floor = 1e-12
        log_v = np.full_like(self.sigma2, np.nan)
        valid_v = np.isfinite(self.sigma2) & (self.sigma2 > 0)
        k_f = self.k.astype(float)
        bias = np.where(k_f > 1, -1.0 / np.maximum(k_f - 1.0, 1.0), 0.0)
        log_v[valid_v] = np.log(np.maximum(self.sigma2[valid_v], floor)) - bias[valid_v]
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
    def mask_bin(self):
        """Rows usable by the BinomialGP — any observation with N >= 1."""
        return self.N >= 1

    @property
    def noise_log_sigma2(self):
        """Approx Var[log s^2] ~= 2/(k-1) (chi-square asymptotic on the
        finite subset used for the variance estimate)."""
        k = self.k.astype(float)
        return np.where(k > 1, 2.0 / np.maximum(k - 1, 1.0), np.nan)
