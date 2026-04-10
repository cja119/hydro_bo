"""
Bayesian Optimiser using GPJax with LBFGS hyperparameter optimisation
and Sobol quasi-random initial sampling.
"""

from abc import ABC, abstractmethod
from typing import Callable, Sequence

import numpy as np
from scipy.stats.qmc import Sobol

from algs.logging_config import get_logger

logger = get_logger(__name__)

def _jax():
    """Lazy accessor for jax — import on first call."""
    import jax
    jax.config.update("jax_enable_x64", True)
    return jax

def _jnp():
    import jax.numpy as jnp
    return jnp

def _gpx():
    import gpjax as gpx
    return gpx

class GaussianProcessRegressor:
    def __init__(self):
        self.posterior = None
        self.data = None
        self._prior = None  

    def _build_prior(self):
        gpx = _gpx()
        return gpx.gps.Prior(
            mean_function=gpx.mean_functions.Constant(),
            kernel=gpx.kernels.RBF(),
        )

    def fit(self, X, y) -> None:
        gpx = _gpx()
        logger.debug("gp_fit_start", n_datapoints=int(X.shape[0]), n_dims=int(X.shape[1]))
        if self._prior is None:
            self._prior = self._build_prior()
        self.data = gpx.Dataset(X=X, y=y)
        likelihood = gpx.likelihoods.Gaussian(num_datapoints=X.shape[0])
        posterior = self._prior * likelihood
        objective = gpx.objectives.ConjugateMLL(negative=True)
        posterior, loss = gpx.fit_lbfgs(
            model=posterior,
            objective=objective,
            train_data=self.data,
            max_iters=200,
        )
        self.posterior = posterior
        logger.debug("gp_fit_complete", final_mll_loss=float(loss))

    def predict(self, X_test):
        """Returns (mean, std) at X_test."""
        dist = self.posterior(X_test, self.data)
        return dist.mean(), dist.stddev()

class AcquisitionFunction(ABC):
    def __init__(self, gp: GaussianProcessRegressor, n_starts: int):
        self.gp = gp
        self.n_starts = n_starts

    @abstractmethod
    def evaluate(self, x) -> None:
        ...

class ExpectedImprovement(AcquisitionFunction):
    def __init__(self, gp: GaussianProcessRegressor, n_starts: int, y_best: float):
        super().__init__(gp, n_starts)
        self.y_best = y_best

    def evaluate(self, x):
        jnp = _jnp()
        if x.ndim == 1:
            x = x[None, :]
        mean, std = self.gp.predict(x)
        std = jnp.clip(std, 1e-9, None)
        z = (mean - self.y_best) / std
        from jax.scipy.stats.norm import cdf, pdf
        ei = std * (z * cdf(z) + pdf(z))
        return ei.squeeze()
    
def sobol_sample(bounds: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Draw n quasi-random points from bounds using Sobol sequence."""
    d = bounds.shape[0]
    sampler = Sobol(d=d, scramble=True, seed=seed)
    unit = sampler.random(n)                  # (n, d) in [0, 1)
    lo, hi = bounds[:, 0], bounds[:, 1]
    return lo + unit * (hi - lo)             # scale to bounds

class BayesianOptimizer:
    def __init__(
        self,
        f: Callable,
        bounds: Sequence,
        n_initial_points: int,
        iter_limit: int,
        n_restarts: int = 5,
        seed: int = 0,
    ):
        self.f = f
        self.bounds = np.asarray(bounds, dtype=float)
        self.n_initial_points = n_initial_points
        self.iter_limit = iter_limit
        self.n_restarts = n_restarts
        self.seed = seed

        self.gp = GaussianProcessRegressor()
        self._X: list[np.ndarray] = []
        self._y: list[float] = []

    def run(self) -> tuple[np.ndarray, float]:
        """Run full BO loop. Returns (best_x, best_y)."""

        logger.info("bo_sobol_phase_start", n_initial_points=self.n_initial_points)
        X_init = sobol_sample(self.bounds, self.n_initial_points, seed=self.seed)
        for i, x in enumerate(X_init):
            y = self.f(x)
            self._X.append(np.asarray(x, dtype=float))
            self._y.append(float(y))
            logger.info("bo_sobol_observation", point=i + 1, n_total=self.n_initial_points, score=float(y))

        logger.info("bo_phase_start", iter_limit=self.iter_limit)
        for i in range(self.iter_limit):
            self._fit_gp()
            x_next = self._suggest()
            y_next = self.f(x_next)
            self._X.append(np.asarray(x_next, dtype=float))
            self._y.append(float(y_next))
            best_idx = int(np.argmax(self._y))
            logger.info(
                "bo_iteration",
                iteration=i + 1,
                iter_limit=self.iter_limit,
                score=float(y_next),
                best_score=float(self._y[best_idx]),
            )

        best_idx = int(np.argmax(self._y))
        logger.info("bo_complete", best_score=float(self._y[best_idx]), n_total_evals=len(self._y))
        return self._X[best_idx], self._y[best_idx]

    def suggest(self) -> np.ndarray:
        """Suggest next point (requires at least one observation)."""
        self._fit_gp()
        return self._suggest()

    def observe(self, x: np.ndarray, y: float) -> None:
        """Manually record an observation."""
        self._X.append(np.asarray(x, dtype=float))
        self._y.append(float(y))

    def _observe(self, x: np.ndarray) -> float:
        y = self.f(x)
        self._X.append(np.asarray(x, dtype=float))
        self._y.append(float(y))
        return float(y)

    def _fit_gp(self) -> None:
        jnp = _jnp()
        logger.debug("bo_gp_refit", n_observations=len(self._y))
        X = jnp.array(np.stack(self._X), dtype=jnp.float64)
        y = jnp.array(self._y, dtype=jnp.float64).reshape(-1, 1)
        self.gp.fit(X, y)

    def _suggest(self) -> np.ndarray:
        y_best = float(max(self._y))
        logger.debug("bo_acquisition_optimise", y_best=y_best, n_restarts=self.n_restarts)
        acq = ExpectedImprovement(self.gp, n_starts=self.n_restarts, y_best=y_best)

        candidates = sobol_sample(
            self.bounds, self.n_restarts, seed=self.seed + len(self._y)
        )

        best_x, best_ei = None, -np.inf
        for x0 in candidates:
            result = _lbfgs_maximise_acquisition(acq.evaluate, x0, self.bounds)
            if result[1] > best_ei:
                best_x, best_ei = result

        logger.debug("bo_acquisition_optimise_complete", best_ei=float(best_ei))
        return best_x

def _lbfgs_maximise_acquisition(
    acq_fn: Callable,
    x0: np.ndarray,
    bounds: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Minimise -acq_fn using jaxopt LBFGS with box-projection at each step."""
    from jaxopt import LBFGS

    jnp = _jnp()
    lo = jnp.array(bounds[:, 0], dtype=jnp.float64)
    hi = jnp.array(bounds[:, 1], dtype=jnp.float64)

    def neg_acq(x):
        x_clipped = jnp.clip(x, lo, hi)
        return -acq_fn(x_clipped)

    solver = LBFGS(fun=neg_acq, maxiter=200)
    result = solver.run(jnp.array(x0, dtype=jnp.float64))
    x_opt = np.clip(np.array(result.params), bounds[:, 0], bounds[:, 1])
    ei_val = float(-result.state.value)
    if not np.isfinite(ei_val):
        logger.warning("lbfgs_acquisition_non_finite", ei_val=ei_val)
    return x_opt, ei_val
