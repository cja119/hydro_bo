"""Parametric BO over the joint space [x_design | theta].

The GP is fit on the joint input, so every evaluation informs the whole
theta-space through the kernel. Theta is *not* maximised by the
acquisition — that would hunt for the single best theta rather than map
how the optimal design varies with it. Instead each iteration draws a
theta node (Sobol over the theta box) and optimises the acquisition over
x with that theta frozen.

Freezing reuses the solver's categorical mechanism: `_layout` reports the
theta dims alongside the integer dims, so their values ride into the
objective closure and only the continuous design dims stay in the SQP
decision vector. Theta dims are kept out of `round_info`, so the kernel
never grid-snaps them.
"""

from __future__ import annotations

import numpy as np
from scipy.stats.qmc import Sobol

from hydro_bo.opt.bayesopt import ConstrainedBayesopt
from hydro_bo.opt.solvers import ConstrainedMixedIntNLP, MixedIntNLP
from hydro_bo.utils.logging_config import get_logger

logger = get_logger(__name__)


class _ThetaFrozenMixin:
    """Pins the theta dims to `theta_unit` (unit-cube coords) by folding
    them into the solver's frozen-dim layout alongside the integer dims."""

    def __init__(self, *args, theta_dims=(), theta_unit=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.theta_dims = tuple(int(i) for i in theta_dims)
        self.theta_unit = tuple(float(v) for v in theta_unit)
        if len(self.theta_dims) != len(self.theta_unit):
            raise ValueError("theta_dims and theta_unit length mismatch")

    def _layout(self):
        cat_idx, combos = super()._layout()
        if not self.theta_dims:
            return cat_idx, combos
        overlap = set(cat_idx) & set(self.theta_dims)
        if overlap:
            raise ValueError(f"theta dims overlap integer dims: {sorted(overlap)}")
        merged = tuple(cat_idx) + self.theta_dims
        order = sorted(range(len(merged)), key=lambda i: merged[i])
        sorted_idx = tuple(merged[i] for i in order)
        out = []
        for combo in combos:
            vals = tuple(combo) + self.theta_unit
            out.append(tuple(vals[i] for i in order))
        return sorted_idx, out


class ThetaFrozenMixedIntNLP(_ThetaFrozenMixin, MixedIntNLP):
    """Unconstrained acquisition solver with theta pinned."""


class ThetaFrozenConstrainedNLP(_ThetaFrozenMixin, ConstrainedMixedIntNLP):
    """Chance-constrained acquisition solver with theta pinned."""


class ParametricBayesopt(ConstrainedBayesopt):
    """ConstrainedBayesopt over [x | theta] with per-iteration theta nodes.

    `bounds` is the joint box; the trailing `d_theta` rows are the theta
    box. `cat_vars` / `round_info` cover only the design integer dims, so
    the joint GP treats theta as ordinary continuous inputs.
    """

    def __init__(self, *args, d_theta: int, theta_seed: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_theta = int(d_theta)
        if self.d_theta <= 0:
            raise ValueError("d_theta must be positive; use ConstrainedBayesopt instead")
        self.d_design = self.bounds.shape[0] - self.d_theta
        self.theta_dims = tuple(range(self.d_design, self.bounds.shape[0]))
        bad = [i for i, _ in self.cat_vars if i >= self.d_design]
        if bad:
            raise ValueError(f"integer dims {bad} fall inside the theta block")
        self._theta_sampler = Sobol(d=self.d_theta, scramble=True, seed=int(theta_seed))

    def next_theta_unit(self) -> np.ndarray:
        """Next Sobol node in the unit theta box. Drawn per iteration so
        the nodes spread over theta-space as the run proceeds."""
        return np.asarray(self._theta_sampler.random(1)[0], dtype=float)

    def theta_to_original(self, theta_unit: np.ndarray) -> np.ndarray:
        lo = self.bounds[self.d_design:, 0]
        hi = self.bounds[self.d_design:, 1]
        return lo + np.asarray(theta_unit, dtype=float) * (hi - lo)

    def _build_solver(self, seed: int):
        theta_unit = self.next_theta_unit()
        logger.info(
            "parametric_theta_node",
            theta=[float(v) for v in self.theta_to_original(theta_unit)],
        )
        common = dict(
            cat_vars=self.cat_vars,
            seed=seed,
            pow_sobol=self.pow_sobol,
            n_restarts=self.n_restarts,
            sqp_config=self.sqp_config,
            theta_dims=self.theta_dims,
            theta_unit=theta_unit,
        )
        if self._stuck_skip:
            return ThetaFrozenMixedIntNLP(**common)
        return ThetaFrozenConstrainedNLP(l1_penalty=self.l1_penalty, **common)
