"""Acquisition functions for the Bayesian optimiser.

Each `AcquisitionFunction` subclass exposes the data the solver layer
needs to optimise it: a jitted negative-acquisition callable for LBFGSB,
a jitted batched callable for the Sobol screen, and a `state_args`
tuple of pytrees that those callables accept after `x` (and any
extra mask args from branch-no-bound). Keeping the contract uniform
means `solvers.py` doesn't have to know which acquisition it's
optimising — adding a new one (e.g. cEI) is just a new class.
"""

from abc import ABC, abstractmethod
from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.stats.norm import cdf, pdf

from hydro_bo.opt.dataset import Dataset
from hydro_bo.opt.surrogate import HeteroscedasticGP, BinomialGP, _predict


# --------------------------------------------------------------------- #
# Pure jitted helpers — keyed on static `round_info`.
# --------------------------------------------------------------------- #


@partial(jax.jit, static_argnames="round_info")
def _ei_g_stats(x, gp_mu_state, gp_log_var_state, scaling, round_info):
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


@partial(jax.jit, static_argnames="round_info")
def _ei_eval(x, gp_mu_state, gp_log_var_state, scaling, g_best, jitter, round_info):
    """EI value at a single point. x has shape (d,)."""
    x2 = x[None, :]
    e_g, var_g = _ei_g_stats(x2, gp_mu_state, gp_log_var_state, scaling, round_info)
    sigma_g = jnp.sqrt(jnp.clip(var_g, jitter, None))
    delta = e_g - g_best
    z = delta / sigma_g
    ei = delta * cdf(z) + sigma_g * pdf(z)
    return ei.squeeze()


@partial(jax.jit, static_argnames="round_info")
def _ei_neg(x, gp_mu_state, gp_log_var_state, scaling, g_best, jitter, round_info):
    return -_ei_eval(
        x, gp_mu_state, gp_log_var_state, scaling, g_best, jitter, round_info
    )


@partial(jax.jit, static_argnames="round_info")
def _ei_batch(
    x_batch, gp_mu_state, gp_log_var_state, scaling, g_best, jitter, round_info
):
    return jax.vmap(_ei_eval, in_axes=(0, None, None, None, None, None, None))(
        x_batch, gp_mu_state, gp_log_var_state, scaling, g_best, jitter, round_info
    )


@partial(jax.jit, static_argnames=("mask_indices", "round_info"))
def _ei_neg_masked(
    x_reduced,
    mask_values,
    gp_mu_state,
    gp_log_var_state,
    scaling,
    g_best,
    jitter,
    mask_indices,
    round_info,
):
    """Insert mask_values at mask_indices in x_reduced, then evaluate
    -EI. mask_indices is static (one trace per index pattern);
    mask_values is a JAX array, so combos share the same compile."""
    x = x_reduced
    for i, idx in enumerate(mask_indices):
        left = x[..., :idx]
        right = x[..., idx:]
        fill = jnp.broadcast_to(mask_values[i].astype(x.dtype), x.shape[:-1] + (1,))
        x = jnp.concatenate([left, fill, right], axis=-1)
    return _ei_neg(
        x, gp_mu_state, gp_log_var_state, scaling, g_best, jitter, round_info
    )


@partial(jax.jit, static_argnames=("mask_indices", "round_info"))
def _ei_batch_masked(
    x_batch,
    mask_values,
    gp_mu_state,
    gp_log_var_state,
    scaling,
    g_best,
    jitter,
    mask_indices,
    round_info,
):
    """Vmapped POSITIVE EI over a batch of x_reduced points."""

    def expand_one(x_red_one):
        x = x_red_one
        for i, idx in enumerate(mask_indices):
            left = x[..., :idx]
            right = x[..., idx:]
            fill = jnp.broadcast_to(mask_values[i].astype(x.dtype), x.shape[:-1] + (1,))
            x = jnp.concatenate([left, fill, right], axis=-1)
        return x

    x_full_batch = jax.vmap(expand_one)(x_batch)
    return jax.vmap(_ei_eval, in_axes=(0, None, None, None, None, None, None))(
        x_full_batch, gp_mu_state, gp_log_var_state, scaling, g_best, jitter, round_info
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


@partial(jax.jit, static_argnames="round_info")
def _feasibility_eval(x, gp_bin_state, log_p_targ, z_sc, round_info):
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
    )
    sigma = jnp.sqrt(jnp.clip(var, 1e-12, None))
    return (mu - z_sc * sigma - log_p_targ).squeeze()


@partial(jax.jit, static_argnames="round_info")
def _feasibility_batch(x_batch, gp_bin_state, log_p_targ, z_sc, round_info):
    """Vmapped LHS over a batch — returns shape (B,)."""
    return jax.vmap(_feasibility_eval, in_axes=(0, None, None, None, None))(
        x_batch, gp_bin_state, log_p_targ, z_sc, round_info
    )


@partial(jax.jit, static_argnames=("mask_indices", "round_info"))
def _feasibility_eval_masked(
    x_reduced, mask_values, gp_bin_state, log_p_targ, z_sc,
    mask_indices, round_info,
):
    """Masked single-point LHS: insert mask_values at mask_indices in
    x_reduced, then evaluate `_feasibility_eval`."""
    x = x_reduced
    for i, idx in enumerate(mask_indices):
        left = x[..., :idx]
        right = x[..., idx:]
        fill = jnp.broadcast_to(mask_values[i].astype(x.dtype), x.shape[:-1] + (1,))
        x = jnp.concatenate([left, fill, right], axis=-1)
    return _feasibility_eval(x, gp_bin_state, log_p_targ, z_sc, round_info)


@partial(jax.jit, static_argnames=("mask_indices", "round_info"))
def _feasibility_batch_masked(
    x_batch, mask_values, gp_bin_state, log_p_targ, z_sc,
    mask_indices, round_info,
):
    """Vmapped masked LHS over a batch of x_reduced points."""
    def expand_one(x_red_one):
        x = x_red_one
        for i, idx in enumerate(mask_indices):
            left = x[..., :idx]
            right = x[..., idx:]
            fill = jnp.broadcast_to(mask_values[i].astype(x.dtype), x.shape[:-1] + (1,))
            x = jnp.concatenate([left, fill, right], axis=-1)
        return x

    x_full_batch = jax.vmap(expand_one)(x_batch)
    return jax.vmap(_feasibility_eval, in_axes=(0, None, None, None, None))(
        x_full_batch, gp_bin_state, log_p_targ, z_sc, round_info
    )


# --------------------------------------------------------------------- #
# Acquisition objects.
# --------------------------------------------------------------------- #


class AcquisitionFunction(ABC):
    """Uniform interface so the solver layer can drive any acquisition.

    Subclasses must populate `state_args` (a tuple of jax pytrees / arrays
    forwarded as the trailing arguments of the jitted callables) and
    expose the four jitted callables. The contract for each callable is
    described on the property docstrings below.

    Constrained subclasses additionally populate `feasibility_state_args`
    and supply `feasibility_batch_fn` / `feasibility_batch_masked_fn`,
    which return jitted callables giving P(feasible | x) ∈ [0, 1] over a
    batch. Used by `ConstrainedMixedIntNLP` to apply an L1 hinge penalty
    in the Sobol screen. Unconstrained acquisitions (e.g. EI) leave the
    defaults that raise NotImplementedError.
    """

    state_args: tuple
    feasibility_state_args: tuple = ()

    @abstractmethod
    def evaluate(self, x): ...

    @abstractmethod
    def neg_acq_fn(self):
        """Returns a jitted callable `f(x, *state_args) -> scalar` for LBFGSB."""
        ...

    @abstractmethod
    def acq_batch_fn(self):
        """Returns a jitted callable `f(x_batch, *state_args) -> (B,)` for sobol screen."""
        ...

    @abstractmethod
    def neg_acq_masked_fn(self, mask_indices: tuple):
        """Returns a jitted callable
        `f(x_red, mask_values, *state_args) -> scalar` for branch-no-bound LBFGSB."""
        ...

    @abstractmethod
    def acq_batch_masked_fn(self, mask_indices: tuple):
        """Returns a jitted callable
        `f(x_batch_red, mask_values, *state_args) -> (B,)` for branch-no-bound sobol."""
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

    def feasibility_eval_masked_fn(self, mask_indices: tuple):
        """Returns a jitted callable
        `f(x_red, mask_values, *feasibility_state_args) -> scalar` for
        the branch-no-bound case. Default raises."""
        raise NotImplementedError(
            f"{type(self).__name__} has no feasibility model; this is an "
            "unconstrained acquisition."
        )

    def feasibility_batch_masked_fn(self, mask_indices: tuple):
        """Returns a jitted callable
        `f(x_batch_red, mask_values, *feasibility_state_args) -> (B,)`
        for the branch-no-bound case. Default raises."""
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
        self.round_info = tuple(round_info)
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
        )
        return jnp.max(e_g)

    def evaluate(self, x):
        return _ei_eval(x, *self.state_args, self.round_info)

    def neg_acq_fn(self):
        return partial(_ei_neg, round_info=self.round_info)

    def acq_batch_fn(self):
        return partial(_ei_batch, round_info=self.round_info)

    def neg_acq_masked_fn(self, mask_indices: tuple):
        return partial(
            _ei_neg_masked, mask_indices=mask_indices, round_info=self.round_info
        )

    def acq_batch_masked_fn(self, mask_indices: tuple):
        return partial(
            _ei_batch_masked, mask_indices=mask_indices, round_info=self.round_info
        )


class ConstrainedExpectedImprovement(ExpectedImprovement):
    """Standard EI on the mu/log-var GPs, plus a chance-bound feasibility
    constraint `f(x) + z_sc·σ_f(x) ≥ log_p_targ` on the BinomialGP latent.

    Callers pass `p_targ` as a probability (gets logit-transformed to the
    latent threshold) and `z_sc` as a standard-normal quantile (z-score)
    directly — no more icdf inside the callable.

    `feasibility_state_args = ()`: the four feasibility callables are
    fully partial-filled with `(gp_bin_state, log_p_targ, z_sc, round_info)`
    at construction time, so the solver layer just calls them with `x`
    (and `mask_values` for the masked variants).
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
        )

    def feasibility_batch_fn(self):
        return partial(
            _feasibility_batch,
            gp_bin_state=self.gp_bin_state,
            log_p_targ=self.log_p_targ,
            z_sc=self.z_sc,
            round_info=self.round_info,
        )

    def feasibility_eval_masked_fn(self, mask_indices: tuple):
        return partial(
            _feasibility_eval_masked,
            gp_bin_state=self.gp_bin_state,
            log_p_targ=self.log_p_targ,
            z_sc=self.z_sc,
            mask_indices=tuple(mask_indices),
            round_info=self.round_info,
        )

    def feasibility_batch_masked_fn(self, mask_indices: tuple):
        return partial(
            _feasibility_batch_masked,
            gp_bin_state=self.gp_bin_state,
            log_p_targ=self.log_p_targ,
            z_sc=self.z_sc,
            mask_indices=tuple(mask_indices),
            round_info=self.round_info,
        )

