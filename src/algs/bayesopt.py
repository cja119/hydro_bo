
from abc import ABC, abstractmethod
from typing import override
import gpjax as gpx
from jaxopt import LBFGS

def sample_space():
    return None


class GaussianProcessRegressor:
    def __init__(self):
        self._cov = gpx.kernels.RBF()
        self._mean = gpx.mean_functions.Constant()
        self._prior = gpx.gps.Prior(mean_function=self._mean, kernel=self._cov)
    
    def fit(self, X, y):
        data = gpx.Dataset(X=X, Y=y)
        liklihood = gpx.likelihoods.Gaussian(num_datapoints=X.shape[0])
        posterior = self._prior * liklihood
        opt_post = gpx.optimiser


class BayesianOptimizer:
    def __init__(self, f, bounds, n_initial_points, iter_limit):
        self.f = f
        self.bounds = bounds
        self.n_initial_points = n_initial_points
        self.iter_limit = iter_limit
        self.gp = GaussianProcessRegressor()

    def suggest(self):
        return None

    def train(self, X, y):
        self.gp.fit(X, y)

class AcquisitionFunction(ABC):
    def __init__(self, gp, n_starts, optimizer):
        self.gp = gp
        self.n_starts = n_starts

    @abstractmethod
    def evaluate(self, x):
        return None

class ExpectedImprovement(AcquisitionFunction):
    def __init__(self, gp, n_starts):
        super().__init__(gp, n_starts)

    @override
    def evaluate(self, x):
        return None


    
