"""Acquisition functions for the Bayesian optimiser.

Each `AcquisitionFunction` subclass exposes jitted callables over the
full input vector plus a `state_args` tuple those callables accept
after `x`. Integer combos are inserted by the solver layer before the
call, so there are no masked variants here.
"""

from abc import ABC, abstractmethod
from functools import partial
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss
from jax.scipy.stats.norm import cdf, pdf

from hydro_bo.opt.dataset import Dataset
from hydro_bo.opt.surrogate import HeteroscedasticGP, BinomialGP, _predict, _predict_covariance
from hydro_bo.opt.solvers import (
    DEFAULT_SQP_CONFIG, MixedIntNLP, _insert, combo_grid, mip_solve, sobol_cloud,
)


# --------------------------------------------------------------------- #
# Pure jitted helpers — keyed on static `round_info`.
# --------------------------------------------------------------------- #


def _build_mu(gp_mu_state, scaling, round_info, mu_kernel_kind):
    """Returns a jitted callable mu(X) -> (mean, var) in objective units.

    Not jitted itself: a jitted function cannot return a Python callable.
    """
    mu_y, sigma_y, _, _, _ = scaling

    def mu_and_var(X):
        mu_s, var_mu_s = _predict(
            gp_mu_state["params"], gp_mu_state["X"], gp_mu_state["L"],
            gp_mu_state["alpha"], gp_mu_state["mean"], gp_mu_state["mask"],
            X, round_info, mu_kernel_kind,
        )
        return mu_s * sigma_y + mu_y, var_mu_s * sigma_y**2

    return jax.jit(mu_and_var)


def _build_beta(gp_mu_state, gp_log_var_state, scaling, round_info,
                lv_kernel_kind, mu_kernel_kind):
    """Returns a jitted callable beta(X_prime, x_cand, var_cand) -> (n,).

        beta = k_n((x',theta'), (x,theta)) / sqrt(sigma_f^2 + sigma_eps^2)

    Numerator is the *posterior* covariance between the prime points and
    the candidate; both denominator terms are evaluated at the CANDIDATE,
    which is where the hypothetical observation is taken. The
    heteroscedastic sigma_eps^2 comes from the log-variance GP.
    """
    _, sigma_y, mu_lv, sigma_lv, _ = scaling

    def beta(X_prime, x_cand, var_cand):
        lv_s, var_lv_s = _predict(
            gp_log_var_state["params"], gp_log_var_state["X"],
            gp_log_var_state["L"], gp_log_var_state["alpha"],
            gp_log_var_state["mean"], gp_log_var_state["mask"],
            x_cand[None, :], round_info, lv_kernel_kind,
        )
        m_lv = lv_s[0] * sigma_lv + mu_lv
        v_lv = var_lv_s[0] * sigma_lv**2
        noise = jnp.exp(m_lv + 0.5 * v_lv)          # E[exp(log var)]

        cov = _predict_covariance(
            gp_mu_state["params"], gp_mu_state["X"], gp_mu_state["L"],
            gp_mu_state["mask"], X_prime, x_cand, round_info, mu_kernel_kind,
        ) * sigma_y**2

        return cov / jnp.sqrt(jnp.clip(var_cand + noise, 1e-12, None))

    return jax.jit(beta)


def _kg_prime_points(x_new, quad_points):
    """Broadcast a design x' against the theta' quadrature nodes."""
    q = quad_points.shape[0]
    return jnp.concatenate(
        [jnp.broadcast_to(x_new, (q, x_new.shape[-1])), quad_points], axis=-1,
    )


def _kg_inner(x_new, x_cand, var_cand, mu, beta, z, quad_points):
    """mu_{n+1}(x', theta' | z, cand) at every theta' node.

    mu_n is evaluated at the PRIME point (x', theta'); beta carries the
    candidate dependence.
    """
    prime = _kg_prime_points(x_new, quad_points)
    mu_prime, _ = mu(prime)
    return mu_prime + beta(prime, x_cand, var_cand) * z


def _theta_integral(x_new, x_cand, var_cand, mu, beta, z, quad_points, quad_weights):
    """E_theta'[mu_{n+1}] = A(x') + z * B(x').

    A and B are separate weighted sums, so the z dependence is affine and
    only the argmax over x' varies with z.
    """
    prime = _kg_prime_points(x_new, quad_points)
    mu_prime, _ = mu(prime)
    A = jnp.sum(quad_weights * mu_prime)
    B = jnp.sum(quad_weights * beta(prime, x_cand, var_cand))
    return A + z * B


def _kg_ab(x_prime_set, x_cand, var_cand, mu, beta, quad_points, quad_weights):
    """(A_i, B_i) over an index set of designs — the `index_set` mode.

    A_i does not depend on the candidate, so it can be hoisted out of the
    acquisition loop; B_i does.
    """
    def one(x_new):
        prime = _kg_prime_points(x_new, quad_points)
        mu_prime, _ = mu(prime)
        return (
            jnp.sum(quad_weights * mu_prime),
            jnp.sum(quad_weights * beta(prime, x_cand, var_cand)),
        )

    return jax.vmap(one)(x_prime_set)


def _kg_index_value(A, B, z):
    """max_i (A_i + z B_i) — the inner max restricted to the index set."""
    return jnp.max(A + z * B)


@partial(
    jax.jit,
    static_argnames=("round_info", "mu_kernel_kind", "lv_kernel_kind"),
)
def _ei_g_stats(
    x, gp_mu_state, gp_log_var_state, scaling, round_info,
    mu_kernel_kind, lv_kernel_kind,
):
    """Mean and variance of g(x) = mu(x) - lam * sigma(x), under the
    moment-matching approximation."""
    mu_y, sigma_y, mu_lv, sigma_lv, lam = scaling

    mu_s, var_mu_s = _predict(
        gp_mu_state["params"],
        gp_mu_state["X"],
        gp_mu_state["L"],
        gp_mu_state["alpha"],
        gp_mu_state["mean"],
        gp_mu_state["mask"],
        x,
        round_info,
        mu_kernel_kind,
    )
    lv_s, var_lv_s = _predict(
        gp_log_var_state["params"],
        gp_log_var_state["X"],
        gp_log_var_state["L"],
        gp_log_var_state["alpha"],
        gp_log_var_state["mean"],
        gp_log_var_state["mask"],
        x,
        round_info,
        lv_kernel_kind,
    )

    m_mu = mu_s * sigma_y + mu_y
    v_mu = var_mu_s * (sigma_y**2)

    m_lv = lv_s * sigma_lv + mu_lv
    v_lv = var_lv_s * (sigma_lv**2)

    e_sd = jnp.exp(0.5 * m_lv + 0.125 * v_lv)
    var_sd = (jnp.exp(0.25 * v_lv) - 1.0) * jnp.exp(m_lv + 0.25 * v_lv)

    e_g = m_mu - lam * e_sd
    var_g = v_mu + (lam**2) * var_sd
    return e_g, var_g


@partial(
    jax.jit,
    static_argnames=("round_info", "mu_kernel_kind", "lv_kernel_kind"),
)
def _ei_eval(
    x, gp_mu_state, gp_log_var_state, scaling, g_best, jitter, round_info,
    mu_kernel_kind, lv_kernel_kind,
):
    """EI value at a single point. x has shape (d,)."""
    x2 = x[None, :]
    e_g, var_g = _ei_g_stats(
        x2, gp_mu_state, gp_log_var_state, scaling, round_info,
        mu_kernel_kind, lv_kernel_kind,
    )
    sigma_g = jnp.sqrt(jnp.clip(var_g, jitter, None))
    delta = e_g - g_best
    z = delta / sigma_g
    ei = delta * cdf(z) + sigma_g * pdf(z)
    return ei.squeeze()


@partial(
    jax.jit,
    static_argnames=("round_info", "mu_kernel_kind", "lv_kernel_kind"),
)
def _ei_neg(
    x, gp_mu_state, gp_log_var_state, scaling, g_best, jitter, round_info,
    mu_kernel_kind, lv_kernel_kind,
):
    return -_ei_eval(
        x, gp_mu_state, gp_log_var_state, scaling, g_best, jitter, round_info,
        mu_kernel_kind, lv_kernel_kind,
    )


@partial(
    jax.jit,
    static_argnames=("round_info", "mu_kernel_kind", "lv_kernel_kind"),
)
def _ei_batch(
    x_batch, gp_mu_state, gp_log_var_state, scaling, g_best, jitter, round_info,
    mu_kernel_kind, lv_kernel_kind,
):
    return jax.vmap(
        _ei_eval, in_axes=(0, None, None, None, None, None, None, None, None),
    )(
        x_batch, gp_mu_state, gp_log_var_state, scaling, g_best, jitter, round_info,
        mu_kernel_kind, lv_kernel_kind,
    )


# --------------------------------------------------------------------- #
# Feasibility-constraint LHS — chance-bound in latent space.
#
# The PG-augmented BinomialGP gives a posterior on the latent f ~ N(μ, σ²);
# probability of feasibility is σ_link(f) ∈ [0,1]. Rather than push f
# through the link inside the SQP constraint (non-affine in f, harms SQP
# convergence) we evaluate the chance bound in latent space directly:
#
#     g(x) = f(x) + Φ^{-1}(z_sc) · √k(x) − log_p_targ
#
# where `log_p_targ` is the user-transformed feasibility threshold and
# `z_sc` is the safety quantile. `g(x) ≥ 0` is the SQP feasibility
# constraint; the same callable batched is consumed by the L1 hinge in
# the Sobol screen via `feasibility_batch_fn`.
# --------------------------------------------------------------------- #


@partial(jax.jit, static_argnames=("round_info", "bin_kernel_kind"))
def _feasibility_eval(x, gp_bin_state, log_p_targ, z_sc, round_info, bin_kernel_kind):
    """LHS of the latent-space chance bound at a single point. `z_sc`
    is the standard-normal quantile (already a z-score, not a
    probability). x has shape (d,); returns scalar."""
    mu, var = _predict(
        gp_bin_state["params"],
        gp_bin_state["X"],
        gp_bin_state["L"],
        gp_bin_state["alpha"],
        gp_bin_state["mean"],
        gp_bin_state["mask"],
        x[None, :],
        round_info,
        bin_kernel_kind,
    )
    sigma = jnp.sqrt(jnp.clip(var, 1e-12, None))
    return (mu - z_sc * sigma - log_p_targ).squeeze()


@partial(jax.jit, static_argnames=("round_info", "bin_kernel_kind"))
def _feasibility_batch(
    x_batch, gp_bin_state, log_p_targ, z_sc, round_info, bin_kernel_kind,
):
    """Vmapped LHS over a batch — returns shape (B,)."""
    return jax.vmap(_feasibility_eval, in_axes=(0, None, None, None, None, None))(
        x_batch, gp_bin_state, log_p_targ, z_sc, round_info, bin_kernel_kind,
    )


# --------------------------------------------------------------------- #
# Acquisition objects.
# --------------------------------------------------------------------- #


class AcquisitionFunction(ABC):
    """Uniform interface so the solver layer can drive any acquisition.

    Subclasses must populate `state_args` (a tuple of jax pytrees / arrays
    forwarded as the trailing arguments of the jitted callables) and
    expose `neg_acq_fn` / `acq_batch_fn`.

    Constrained subclasses additionally populate `feasibility_state_args`
    and supply the feasibility callables; unconstrained acquisitions
    leave the defaults that raise NotImplementedError.
    """

    state_args: tuple
    feasibility_state_args: tuple = ()

    @abstractmethod
    def evaluate(self, x): ...

    @abstractmethod
    def neg_acq_fn(self):
        """Returns a jitted callable `f(x, *state_args) -> scalar`."""
        ...

    @abstractmethod
    def acq_batch_fn(self):
        """Returns a jitted callable `f(x_batch, *state_args) -> (B,)` for the sobol screen."""
        ...

    def feasibility_batch_fn(self):
        """Returns a jitted callable
        `f(x_batch, *feasibility_state_args) -> (B,)` giving
        P(feasible | x). Default raises — only constrained acquisitions
        provide this."""
        raise NotImplementedError(
            f"{type(self).__name__} has no feasibility model; this is an "
            "unconstrained acquisition."
        )

    def feasibility_eval_fn(self):
        """Returns a jitted callable
        `f(x, *feasibility_state_args) -> scalar` giving P(feasible | x)
        at a single point. Used by the SQP layer to express feasibility
        as an explicit inequality constraint
        `p_targ <= P(feasible | x) <= 1`. Default raises."""
        raise NotImplementedError(
            f"{type(self).__name__} has no feasibility model; this is an "
            "unconstrained acquisition."
        )


class ExpectedImprovement(AcquisitionFunction):
    """Approximate EI for a heteroscedastic GP objective.

    The objective is g(x) = mu(x) - lam * sigma(x) with sigma(x) ~
    LogNormal. We assume g(x) is approximately normal and match first
    and second order moments (assuming independence of mu and sigma).
    The acquisition is then EI w.r.t. the incumbent
    g_best = max_i E[g(x_i)] over the existing training inputs.
    """

    def __init__(
        self,
        gp_mu: HeteroscedasticGP,
        gp_log_var: HeteroscedasticGP,
        lam: float,
        dataset: Dataset,
        jitter: float = 1e-9,
        round_info: tuple = (),
    ):
        # round_info must match the round_info both GPs were fit with.
        # Each GP's `kernel_kind` is captured at construction time and
        # baked into the jitted callables via `partial` — JAX uses it
        # as a static argument, so changing it triggers a re-trace.
        self.round_info = tuple(round_info)
        self.mu_kernel_kind = str(gp_mu.kernel_kind)
        self.lv_kernel_kind = str(gp_log_var.kernel_kind)
        self.scaling = (
            jnp.asarray(dataset._mu_y, dtype=jnp.float64),
            jnp.asarray(dataset._sigma_y, dtype=jnp.float64),
            jnp.asarray(dataset._mu_lv, dtype=jnp.float64),
            jnp.asarray(dataset._sigma_lv, dtype=jnp.float64),
            jnp.asarray(lam, dtype=jnp.float64),
        )
        self.jitter = jnp.asarray(jitter, dtype=jnp.float64)
        self.gp_mu_state = gp_mu.state()
        self.gp_log_var_state = gp_log_var.state()
        self._g_best = self._incumbent(dataset)
        self.state_args = (
            self.gp_mu_state,
            self.gp_log_var_state,
            self.scaling,
            self._g_best,
            self.jitter,
        )

    def _incumbent(self, dataset: Dataset):
        m = dataset.mask_mu
        if int(m.sum()) == 0:
            return jnp.asarray(0.0, dtype=jnp.float64)
        X_unit = jnp.asarray(dataset.X_scaled[m], dtype=jnp.float64)
        e_g, _ = _ei_g_stats(
            X_unit,
            self.gp_mu_state,
            self.gp_log_var_state,
            self.scaling,
            self.round_info,
            self.mu_kernel_kind,
            self.lv_kernel_kind,
        )
        return jnp.max(e_g)

    def evaluate(self, x):
        return _ei_eval(
            x, *self.state_args, self.round_info,
            self.mu_kernel_kind, self.lv_kernel_kind,
        )

    def neg_acq_fn(self):
        return partial(
            _ei_neg, round_info=self.round_info,
            mu_kernel_kind=self.mu_kernel_kind, lv_kernel_kind=self.lv_kernel_kind,
        )

    def acq_batch_fn(self):
        return partial(
            _ei_batch, round_info=self.round_info,
            mu_kernel_kind=self.mu_kernel_kind, lv_kernel_kind=self.lv_kernel_kind,
        )


class ConstrainedExpectedImprovement(ExpectedImprovement):
    """Standard EI on the mu/log-var GPs, plus a chance-bound feasibility
    constraint `f(x) + z_sc·σ_f(x) ≥ log_p_targ` on the BinomialGP latent.

    Callers pass `p_targ` as a probability (gets logit-transformed to the
    latent threshold) and `z_sc` as a standard-normal quantile (z-score)
    directly — no more icdf inside the callable.

    `feasibility_state_args = ()`: the feasibility callables are fully
    partial-filled with `(gp_bin_state, log_p_targ, z_sc, round_info)`
    at construction time, so the solver layer just calls them with `x`.
    """

    feasibility_state_args: tuple = ()

    def __init__(
        self,
        gp_mu: HeteroscedasticGP,
        gp_log_var: HeteroscedasticGP,
        gp_bin: BinomialGP,
        lam: float,
        p_targ: float,
        z_sc: float,
        dataset: Dataset,
        jitter: float = 1e-9,
        round_info: tuple = (),
    ):
        super().__init__(
            gp_mu=gp_mu,
            gp_log_var=gp_log_var,
            lam=lam,
            dataset=dataset,
            jitter=jitter,
            round_info=round_info,
        )
        self.gp_bin_state = gp_bin.state()
        self.bin_kernel_kind = str(gp_bin.kernel_kind)
        p = jnp.asarray(p_targ, dtype=jnp.float64)
        self.log_p_targ = jnp.log(p / (1.0 - p))
        self.z_sc = jnp.asarray(z_sc, dtype=jnp.float64)

    def feasibility_eval_fn(self):
        return partial(
            _feasibility_eval,
            gp_bin_state=self.gp_bin_state,
            log_p_targ=self.log_p_targ,
            z_sc=self.z_sc,
            round_info=self.round_info,
            bin_kernel_kind=self.bin_kernel_kind,
        )

    def feasibility_batch_fn(self):
        return partial(
            _feasibility_batch,
            gp_bin_state=self.gp_bin_state,
            log_p_targ=self.log_p_targ,
            z_sc=self.z_sc,
            round_info=self.round_info,
            bin_kernel_kind=self.bin_kernel_kind,
        )


class KnowledgeGradientInner:
    """Holder for the theta' quadrature and the inner objective.

    No longer an `AcquisitionFunction`: the inner problem is solved
    *inside* a trace (once per Hermite node), whereas the
    `AcquisitionFunction` + `maximise` contract assumes one objective per
    BO iteration driven from the host. `KnowledgeGradient` below remains
    an `AcquisitionFunction` and is driven by `MixedIntNLP` as before.
    """

    def __init__(
        self,
        mu: Callable,
        beta: Callable,
        round_info: tuple = (),
        quad_per_dim: int = 5,
        num_theta: int = 0,
    ):
        self.mu = mu
        self.beta = beta
        self.round_info = tuple(round_info)
        self.quad_per_dim = int(quad_per_dim)
        self.num_theta = int(num_theta)
        self._theta_vals, self._quad_weights = self._quad_points(
            self.num_theta, self.quad_per_dim
        )

    def value(self, x_new, x_cand, var_cand, z):
        """E_theta'[mu_{n+1}(x', theta' | z)] — pure, z passed explicitly."""
        return _theta_integral(
            x_new, x_cand, var_cand, self.mu, self.beta, z,
            self._theta_vals, self._quad_weights,
        )

    def ab(self, x_prime_set, x_cand, var_cand):
        return _kg_ab(
            x_prime_set, x_cand, var_cand, self.mu, self.beta,
            self._theta_vals, self._quad_weights,
        )

    def _quad_points(self, dim: int, quad_per_dim: int):
        """Tensor-product Gauss-Legendre on [0,1]^dim through the
        triangular inverse CDF. Weights sum to 1.

        Gauss-Legendre lives on [-1,1]; mapping to [0,1] is u=(t+1)/2 with
        weights halved for the Jacobian.
        """
        if dim == 0:
            return (
                jnp.zeros((1, 0), dtype=jnp.float64),
                jnp.ones(1, dtype=jnp.float64),
            )
        t, w = leggauss(int(quad_per_dim))
        u_1d = 0.5 * (t + 1.0)
        w_1d = 0.5 * w

        grids = np.meshgrid(*([u_1d] * dim), indexing="ij")
        u = np.stack([g.reshape(-1) for g in grids], axis=-1)
        wg = np.meshgrid(*([w_1d] * dim), indexing="ij")
        weights = np.prod(np.stack([g.reshape(-1) for g in wg], axis=-1), axis=-1)

        return (
            self._theta_from_quad(jnp.asarray(u, dtype=jnp.float64)),
            jnp.asarray(weights, dtype=jnp.float64),
        )

    def _theta_from_quad(self, quad_points):
        """Map quadrature points from [0,1]^dim to triangular theta space."""
        return self._inv_cdf(quad_points)

    @staticmethod
    def _inv_cdf(p):
        """Inverse CDF of symmetric triangular distribution.

        Applied in standardised space: theta is scaled by
        (theta - lo)/(hi - lo), which is the triangular's own support, and
        an affine map carries a symmetric triangular on [lo,hi] to one on
        [0,1]. So the physical bounds cancel and never enter here.
        """
        return jnp.where(p < 0.5, jnp.sqrt(p / 2), 1 - jnp.sqrt((1 - p) / 2))


def _outer_integral(x_cand, inner_obj, z_nodes, z_weights, solve_inner):
    """E_z[ max_x' E_theta'[mu_{n+1}] ] — `strict` mode.

    The max sits inside E_z: one inner solve per Hermite node, since the
    argmax moves with z (that movement is exactly what KG measures). The
    z reduction is the weighted sum afterwards.

    Inner maximisers are frozen with `stop_gradient` before the value is
    recomputed, so differentiating this gives the envelope (Danskin)
    gradient without unrolling the inner SQP scans.
    """
    _, var_c = inner_obj.mu(x_cand[None, :])
    var_cand = var_c[0]

    x_stars = jax.vmap(lambda z: solve_inner(x_cand, var_cand, z))(z_nodes)
    x_stars = jax.lax.stop_gradient(x_stars)

    vals = jax.vmap(
        lambda xs, z: inner_obj.value(xs, x_cand, var_cand, z)
    )(x_stars, z_nodes)
    return jnp.sum(z_weights * vals)


def _outer_integral_index(x_cand, inner_obj, z_nodes, z_weights, x_prime_set):
    """E_z[ max_i (A_i + z B_i) ] — `index_set` mode.

    A and B are evaluated once per index-set entry, then reused across
    every Hermite node. Integer dims are attributes of the index entries,
    so no combo enumeration appears here.
    """
    _, var_c = inner_obj.mu(x_cand[None, :])
    var_cand = var_c[0]
    A, B = inner_obj.ab(x_prime_set, x_cand, var_cand)
    vals = jax.vmap(lambda z: _kg_index_value(A, B, z))(z_nodes)
    return jnp.sum(z_weights * vals)


class KnowledgeGradient(AcquisitionFunction):
    """Parametric knowledge gradient over the joint space [x | theta].

        KG(x,theta) = E_z[ max_x' E_theta'[ mu_{n+1}(x',theta' | z,x,theta) ] ]

    Implements the `AcquisitionFunction` contract, so the existing
    `MixedIntNLP` driver maximises it exactly as it does EI.

    `mode`:
      "strict"     — a real inner maximisation per Hermite node, via the
                     traceable `mip_solve` (integer dims enumerated).
      "index_set"  — inner max restricted to a fixed index set of designs,
                     reusing (A_i, B_i) across z nodes.

    KG uses the plain posterior mean: no `mu - lam*sigma` penalty, which
    would double-count the uncertainty KG already integrates over.
    The constant baseline max_x' E_theta'[mu_n] is not subtracted — it is
    candidate-independent and so leaves the argmax unchanged.
    """

    def __init__(
        self,
        gp_mu: HeteroscedasticGP,
        gp_log_var: HeteroscedasticGP,
        dataset: Dataset,
        d_theta: int,
        jitter: float = 1e-9,
        round_info: tuple = (),
        cat_vars=(),
        kg_args: dict = {},
    ) -> None:

        self.round_info = tuple(round_info)
        self.mu_kernel_kind = str(gp_mu.kernel_kind)
        self.lv_kernel_kind = str(gp_log_var.kernel_kind)
        self.jitter = jnp.asarray(jitter, dtype=jnp.float64)

        self.mode = str(kg_args.get("mode", "strict"))
        self.seed = int(kg_args.get("seed", 0))
        self.quad_per_dim = int(kg_args.get("theta_quad_per_dim", 5))
        self.z_quad_points = int(kg_args.get("z_quad_points", 9))
        self.inner_pow_sobol = int(kg_args.get("inner_pow_sobol", 8))
        self.inner_n_restarts = int(kg_args.get("inner_n_restarts", 1))
        self.index_set_pow = int(kg_args.get("index_set_pow", 8))
        self.sqp_config_inner = kg_args.get("inner_sqp_config", None)

        self.d_total = int(dataset.X_scaled.shape[1])
        self.d_theta = int(d_theta)
        self.d_design = self.d_total - self.d_theta

        # Scaling tuple consumed by _build_mu / _build_beta. lam is carried
        # for signature compatibility only; KG does not use it.
        self.scaling = (
            jnp.asarray(dataset._mu_y, dtype=jnp.float64),
            jnp.asarray(dataset._sigma_y, dtype=jnp.float64),
            jnp.asarray(dataset._mu_lv, dtype=jnp.float64),
            jnp.asarray(dataset._sigma_lv, dtype=jnp.float64),
            jnp.asarray(0.0, dtype=jnp.float64),
        )

        self.gp_mu_state = gp_mu.state()
        self.gp_log_var_state = gp_log_var.state()
        self.state_args = (self.gp_mu_state,)
        # `BaseBayesopt._suggest_next` logs the EI incumbent. KG has no
        # incumbent (it scores information, not improvement over a best),
        # so expose NaN rather than fork the shared BO loop.
        self._g_best = jnp.asarray(jnp.nan, dtype=jnp.float64)

        self.mu = _build_mu(
            self.gp_mu_state, self.scaling, self.round_info, self.mu_kernel_kind,
        )
        self.beta = _build_beta(
            self.gp_mu_state, self.gp_log_var_state, self.scaling,
            self.round_info, self.lv_kernel_kind, self.mu_kernel_kind,
        )
        self._inner_obj = KnowledgeGradientInner(
            self.mu, self.beta, self.round_info, self.quad_per_dim, self.d_theta,
        )

        # Integer dims belong to the design block only.
        self.cat_idx, self.combos = combo_grid(cat_vars)
        bad = [i for i in self.cat_idx if i >= self.d_design]
        if bad:
            raise ValueError(f"integer dims {bad} fall inside the theta block")
        self.d_red = self.d_design - len(self.cat_idx)

        self._z_nodes, self._z_weights = self._gauss_hermite_quadrature(
            n_points=self.z_quad_points,
        )
        self._cloud = sobol_cloud(self.d_red, self.inner_pow_sobol, self.seed)
        self._lb = jnp.zeros(self.d_red, dtype=jnp.float64)
        self._ub = jnp.ones(self.d_red, dtype=jnp.float64)
        self._x_prime_set = sobol_cloud(
            self.d_design, self.index_set_pow, self.seed + 1,
        )
        self._value = jax.jit(self._kg_value)
        self._batch = jax.jit(jax.vmap(self._kg_value))

    # ---- inner solve ----

    def _solve_inner(self, x_cand, var_cand, z):
        """argmax_x' E_theta'[mu_{n+1}] for one Hermite node."""
        def neg(x_full):
            return -self._inner_obj.value(x_full, x_cand, var_cand, z)

        x_best, _ = mip_solve(
            neg, self._cloud, self.combos, self.cat_idx,
            self._lb, self._ub,
            self.sqp_config_inner or DEFAULT_SQP_CONFIG,
            n_restarts=self.inner_n_restarts,
        )
        return x_best

    def _kg_value(self, x_cand):
        if self.mode == "index_set":
            return _outer_integral_index(
                x_cand, self._inner_obj, self._z_nodes, self._z_weights,
                self._x_prime_set,
            )
        return _outer_integral(
            x_cand, self._inner_obj, self._z_nodes, self._z_weights,
            self._solve_inner,
        )

    # ---- AcquisitionFunction contract ----

    def evaluate(self, x_theta_new):
        return self._value(jnp.asarray(x_theta_new, dtype=jnp.float64).ravel())

    def neg_acq_fn(self):
        value = self._value

        def neg(x, *_state):
            return -value(x).reshape(())

        return neg

    def acq_batch_fn(self):
        batch = self._batch

        def f(x_batch, *_state):
            return batch(x_batch)

        return f

    def _gauss_hermite_quadrature(self, n_points: int):
        """Gauss-Hermite for E_{Z~N(0,1)}.

        Physicists' nodes are scaled by sqrt(2) and weights by 1/sqrt(pi);
        the weights then sum to 1. The sqrt(2) belongs on the nodes — it is
        the change of variables, not a transform of the candidate.
        """
        points, weights = hermgauss(int(n_points))
        return (
            jnp.asarray(np.sqrt(2.0) * points, dtype=jnp.float64),
            jnp.asarray(weights / np.sqrt(np.pi), dtype=jnp.float64),
        )
