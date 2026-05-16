"""Subprocess wrapper for `solver.maximise(acq)` with hard wall-clock timeout.

JAX/XLA computations run in C++ and don't reliably honour Python signals,
so a `signal.SIGALRM` or thread-based timeout can fail to interrupt a
genuinely stuck SQP. The only reliable way to kill it is `Process.terminate()`,
which sends SIGTERM to the child PID and brings everything down.

The child receives a fully self-contained `payload` dict (only Python
primitives + numpy arrays — no JAX state, no closures) and rebuilds the
acquisition + solver inside the subprocess. This means the child pays the
XLA compile cost from scratch every call — typically a few seconds on most
backends, dwarfed by the actual SQP execute time.

Public API
----------
`run_maximise_with_timeout(payload, timeout_sec) -> (x_unit, val, status)`
where status is one of:
  - "ok"          : SQP returned a result
  - "timeout"     : child was terminated after timeout_sec
  - "no_feasible" : feasible_screen exhausted; caller should fall back
  - "error"       : child raised some other exception; payload['error'] has it
"""

from __future__ import annotations

import dataclasses
import multiprocessing as mp
import time
import traceback
from typing import Any

import numpy as np


# Multiprocessing context: 'spawn' is the only mode that works reliably with
# JAX on every platform (fork carries the parent's already-imported jax
# state, which can deadlock under XLA's threadpools).
_MP_CTX = mp.get_context("spawn")


def run_maximise_with_timeout(
    payload: dict,
    timeout_sec: int,
) -> tuple[np.ndarray | None, float | None, str, str]:
    """Run `solver.maximise(acq)` in a subprocess with a hard wall-clock cap.

    The child rebuilds the acquisition + solver from `payload` (which must
    contain only picklable primitives — numpy arrays, scalars, tuples). On
    timeout, the child is terminated via SIGTERM.

    Returns
    -------
    (x_unit, val, status, detail)
        x_unit : (D,) numpy array, or None on non-"ok" status.
        val    : best acquisition value, or None.
        status : "ok" | "timeout" | "no_feasible" | "error".
        detail : human-readable info string (empty when status == "ok").
    """
    q: Any = _MP_CTX.Queue(maxsize=1)
    proc = _MP_CTX.Process(target=_worker, args=(payload, q), daemon=True)

    t0 = time.perf_counter()
    proc.start()
    proc.join(timeout=timeout_sec)
    elapsed = time.perf_counter() - t0

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=10)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5)
        return (
            None, None, "timeout",
            f"subprocess exceeded {timeout_sec}s (elapsed={elapsed:.1f}s); killed",
        )

    if q.empty():
        return None, None, "error", f"subprocess exited without result (exitcode={proc.exitcode})"

    result = q.get_nowait()
    status = result.get("status", "error")
    if status == "ok":
        return (
            np.asarray(result["x"], dtype=float),
            float(result["val"]),
            "ok",
            "",
        )
    if status == "no_feasible":
        return None, None, "no_feasible", result.get("detail", "")
    return None, None, "error", result.get("detail", "")


def _worker(payload: dict, q) -> None:
    """Child-process entry point. Must be picklable, top-level."""
    try:
        # Configure logging in the child to APPEND to the same run.log the
        # parent writes to — otherwise every log line emitted by the
        # subprocess (feasible_screen_batch, maximise_timings, ...) lands
        # in the child's stderr instead of the user-visible log file, and
        # we can't see how far solve_batch got when a timeout kicks in.
        # We attach a fresh FileHandler in APPEND mode so the parent's
        # existing log content is preserved.
        log_file = payload.get("log_file")
        if log_file:
            import logging
            import structlog
            from hydro_bo.utils.logging_config import _shared_processors

            shared = _shared_processors()
            structlog.configure(
                processors=shared + [
                    structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
                ],
                wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                cache_logger_on_first_use=False,
            )
            file_formatter = structlog.stdlib.ProcessorFormatter(
                foreign_pre_chain=shared,
                processors=[
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    structlog.dev.ConsoleRenderer(colors=False),
                ],
            )
            fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            fh.setFormatter(file_formatter)
            fh.setLevel(logging.INFO)
            root = logging.getLogger()
            for h in list(root.handlers):
                root.removeHandler(h)
            root.addHandler(fh)
            root.setLevel(logging.INFO)
            logging.getLogger("hydro_bo").setLevel(logging.INFO)

        # Inside the child, this is the first jax import — XLA flags
        # already set by the parent's environment carry over.
        import jax
        import jax.numpy as jnp
        from hydro_bo.opt.acquisition import (
            ExpectedImprovement,
            ConstrainedExpectedImprovement,
        )
        from hydro_bo.opt.solvers import (
            MixedIntNLP,
            ConstrainedMixedIntNLP,
            NoFeasibleScreenError,
        )
        from septal.jax.sqp import SQPConfig

        # ── Rebuild GP states from numpy ────────────────────────────────
        def _to_jax_state(np_state):
            return {
                "X": jnp.asarray(np_state["X"], dtype=jnp.float64),
                "L": jnp.asarray(np_state["L"], dtype=jnp.float64),
                "alpha": jnp.asarray(np_state["alpha"], dtype=jnp.float64),
                "mean": jnp.asarray(np_state["mean"], dtype=jnp.float64),
                "mask": jnp.asarray(np_state["mask"], dtype=jnp.float64),
                "params": {
                    "log_amp": jnp.asarray(np_state["params"]["log_amp"], dtype=jnp.float64),
                    "log_ls": jnp.asarray(np_state["params"]["log_ls"], dtype=jnp.float64),
                    **({"mean": jnp.asarray(np_state["params"]["mean"], dtype=jnp.float64)}
                       if "mean" in np_state["params"] else {}),
                },
            }

        # Acquisition is rebuilt with shims that expose the same `.state()`
        # method the GP objects do. We don't run a GP fit in the child —
        # we already have the fitted state.
        class _StaticGP:
            def __init__(self, np_state, kernel_kind):
                self._state = _to_jax_state(np_state)
                self.round_info = tuple(
                    (int(i), int(n)) for i, n in payload["round_info"]
                )
                self.kernel_kind = kernel_kind
            def state(self):
                return self._state

        gp_mu = _StaticGP(payload["gp_mu"], payload["mu_kernel"])
        gp_lv = _StaticGP(payload["gp_lv"], payload["lv_kernel"])

        # Dataset shim — acquisition only reads _mu_y/_sigma_y/_mu_lv/_sigma_lv,
        # X_scaled (for incumbent), mask_mu.
        class _StaticDataset:
            pass
        ds = _StaticDataset()
        ds._mu_y = float(payload["scaling"]["mu_y"])
        ds._sigma_y = float(payload["scaling"]["sigma_y"])
        ds._mu_lv = float(payload["scaling"]["mu_lv"])
        ds._sigma_lv = float(payload["scaling"]["sigma_lv"])
        ds.X_scaled = np.asarray(payload["dataset_X_scaled"], dtype=float)
        ds.mask_mu = np.asarray(payload["dataset_mask_mu"], dtype=bool)

        if payload["constrained"]:
            gp_bin = _StaticGP(payload["gp_bin"], payload["bin_kernel"])
            acq = ConstrainedExpectedImprovement(
                gp_mu=gp_mu, gp_log_var=gp_lv, gp_bin=gp_bin,
                lam=payload["lam"],
                p_targ=payload["p_targ"], z_sc=payload["z_sc"],
                dataset=ds, jitter=payload["jitter"],
                round_info=tuple((int(i), int(n)) for i, n in payload["round_info"]),
            )
            cfg = SQPConfig(**payload["sqp_config_kwargs"])
            solver = ConstrainedMixedIntNLP(
                cat_vars=payload["cat_vars"],
                l1_penalty=payload["l1_penalty"],
                seed=payload["seed"],
                pow_sobol=payload["pow_sobol"],
                n_restarts=payload["n_restarts"],
                sqp_config=cfg,
                feasible_screen=payload["feasible_screen"],
                max_screen_batches=payload["max_screen_batches"],
            )
        else:
            acq = ExpectedImprovement(
                gp_mu=gp_mu, gp_log_var=gp_lv,
                lam=payload["lam"], dataset=ds,
                jitter=payload["jitter"],
                round_info=tuple((int(i), int(n)) for i, n in payload["round_info"]),
            )
            cfg = SQPConfig(**payload["sqp_config_kwargs"])
            solver = MixedIntNLP(
                cat_vars=payload["cat_vars"],
                seed=payload["seed"],
                pow_sobol=payload["pow_sobol"],
                n_restarts=payload["n_restarts"],
                sqp_config=cfg,
                max_screen_batches=payload["max_screen_batches"],
            )

        try:
            x_unit, val = solver.maximise(acq)
        except NoFeasibleScreenError as e:
            q.put({"status": "no_feasible", "detail": str(e)})
            return

        q.put({
            "status": "ok",
            "x": np.asarray(x_unit, dtype=float),
            "val": float(val),
        })
    except Exception as e:
        q.put({
            "status": "error",
            "detail": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        })


def serialise_acq_payload(
    bo,
    seed: int,
) -> dict:
    """Build the picklable `payload` dict from a fitted BO object.

    Reads from `bo.gp_mu`, `bo.gp_log_var`, (and `bo.gp_bin` if constrained)
    and `bo.dataset`. The numpy conversion happens HERE in the parent so the
    child only sees plain primitives.

    Caller passes the iteration seed (typically `bo.seed + len(bo._X)`)
    so the child's Sobol scramble matches what the in-process path would do.
    """
    def _to_np(jax_state):
        return {
            "X": np.asarray(jax_state["X"]),
            "L": np.asarray(jax_state["L"]),
            "alpha": np.asarray(jax_state["alpha"]),
            "mean": np.asarray(jax_state["mean"]),
            "mask": np.asarray(jax_state["mask"]),
            "params": {
                "log_amp": np.asarray(jax_state["params"]["log_amp"]),
                "log_ls": np.asarray(jax_state["params"]["log_ls"]),
                **({"mean": np.asarray(jax_state["params"]["mean"])}
                   if "mean" in jax_state["params"] else {}),
            },
        }

    is_constrained = hasattr(bo, "gp_bin")
    ds = bo.dataset

    # SQP config → dict of primitives for SQPConfig() ctor.
    sqp_cfg = bo.sqp_config
    sqp_kwargs = {
        f.name: getattr(sqp_cfg, f.name)
        for f in dataclasses.fields(sqp_cfg)
        if isinstance(getattr(sqp_cfg, f.name), (int, float, bool, str))
    }

    # Locate the parent's log-file so the child can append to it. Look for
    # any FileHandler attached to the root logger and grab its filename.
    import logging as _logging
    log_file = None
    for _h in _logging.getLogger().handlers:
        if isinstance(_h, _logging.FileHandler):
            log_file = str(_h.baseFilename)
            break

    payload = {
        "constrained": is_constrained,
        "gp_mu": _to_np(bo.gp_mu.state()),
        "gp_lv": _to_np(bo.gp_log_var.state()),
        "round_info": list(bo.gp_mu.round_info),
        "mu_kernel": bo.gp_mu_kernel,
        "lv_kernel": bo.gp_log_var_kernel,
        "log_file": log_file,
        "scaling": {
            "mu_y": float(ds._mu_y),
            "sigma_y": float(ds._sigma_y),
            "mu_lv": float(ds._mu_lv),
            "sigma_lv": float(ds._sigma_lv),
        },
        "dataset_X_scaled": np.asarray(ds.X_scaled),
        "dataset_mask_mu": np.asarray(ds.mask_mu),
        "lam": float(bo.lam),
        "jitter": 1e-9,
        "cat_vars": [(int(i), [float(v) for v in vals]) for i, vals in bo.cat_vars],
        "seed": int(seed),
        "pow_sobol": int(bo.pow_sobol),
        "n_restarts": int(bo.n_restarts),
        "max_screen_batches": int(bo.max_screen_batches),
        "sqp_config_kwargs": sqp_kwargs,
    }
    if is_constrained:
        payload["gp_bin"] = _to_np(bo.gp_bin.state())
        payload["bin_kernel"] = bo.gp_bin_kernel
        payload["p_targ"] = float(bo.p_targ)
        payload["z_sc"] = float(bo.z_sc)
        payload["l1_penalty"] = float(bo.l1_penalty)
        payload["feasible_screen"] = bool(bo.feasible_screen)
    return payload
