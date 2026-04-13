"""
Dispatcher for multiple MPC solves
"""


import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
import ray
import numpy as np

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])

@contextmanager
def suppress_worker_output():
    with open(os.devnull, "w") as devnull:
        old_out, old_err = sys.stdout, sys.stderr
        gurobi_logger = logging.getLogger("gurobipy")
        old_gurobi_level = gurobi_logger.level
        sys.stdout, sys.stderr = devnull, devnull
        gurobi_logger.setLevel(logging.CRITICAL + 1)
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err
            gurobi_logger.setLevel(old_gurobi_level)

class RayMultiMPC:
    def __init__(self, env_args, param_overrides, num_instances, num_devices, timeout, exit_fraction, dump_diagnostics_on_failure=False):
        self._env_args = env_args
        self._params = param_overrides
        self._num_instances = num_instances
        self._timeout = timeout
        self._exit_fraction = exit_fraction
        self._dump_diagnostics = dump_diagnostics_on_failure

        if ray.is_initialized():
            ray.shutdown()
        ray.init(
            num_cpus=num_devices,
            runtime_env={"env_vars": {"PYTHONPATH": _PROJECT_ROOT}},
            _metrics_export_port=None,  # Disable metrics exporter
            logging_level=logging.ERROR,  # Reduce Ray logging verbosity
        )

    def run_multisim(self):
        import time as _time
        from hydro_bo.algs.logging_config import get_logger
        from collections import Counter

        logger = get_logger(__name__)
        start_time = _time.perf_counter()

        # Launch all instances at once
        tasks = [run_mpc.remote(self._env_args, id=f"iter_{i}", params=self._params, dump_diagnostics=self._dump_diagnostics) for i in range(self._num_instances)]

        if self._dump_diagnostics:
            # Wait for tasks one at a time, exit early on first failure
            results = []
            for i, task in enumerate(tasks):
                result = ray.get([task])[0]
                results.append(result)
                success, _, _ = result
                if not success:
                    logger.info("early_exit_on_failure", task_index=i, reason="Diagnostic dump requested")
                    # Cancel remaining tasks
                    for remaining_task in tasks[i+1:]:
                        ray.cancel(remaining_task)
                    break
        else:
            results = ray.get(tasks)

        # Collect scores and relaxation info
        scores = [score for _, score, _ in results]
        n_success = sum(1 for success, _, _ in results if success)

        # Analyze relaxation usage across all simulations
        all_relaxations = []
        for _, _, relax_info in results:
            if relax_info and 'relaxations_used' in relax_info:
                all_relaxations.extend(relax_info['relaxations_used'])

        relaxation_counts = Counter(all_relaxations)
        elapsed = _time.perf_counter() - start_time

        logger.info("multisim_complete",
                   n_success=n_success,
                   n_total=self._num_instances,
                   elapsed_seconds=elapsed,
                   relaxation_usage=dict(relaxation_counts))

        return scores

@ray.remote
def run_mpc(env_args, id, params, dump_diagnostics=False):
    import structlog
    import warnings
    from hydro_bo import ShippingEnv

    # Silence structlog in worker — it writes directly to stderr via PrintLogger
    # so logging.setLevel() has no effect; redirect the factory to /dev/null.
    _devnull = open(os.devnull, "w")
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(file=_devnull))

    # Suppress all Pyomo warnings in worker processes
    pyomo_logger = logging.getLogger("pyomo.core")
    pyomo_logger.setLevel(logging.CRITICAL + 1)

    warnings.filterwarnings("ignore", category=UserWarning, module="pyomo")
    warnings.filterwarnings("ignore", message=".*Loading a SolverResults.*")
    warnings.filterwarnings("ignore", message=".*termination condition.*")

    # Derive an integer seed from the Ray task ID hex string
    ctx = ray.get_runtime_context()
    seed = int(ctx.get_task_id(), 16) % (2**31)

    def _deep_set(target, updates):
        for k, v in updates.items():
            if isinstance(v, dict):
                if k not in target or not isinstance(target[k], dict):
                    target[k] = {}
                _deep_set(target[k], v)
            else:
                target[k] = v

    env = ShippingEnv('shipping')
    with env as e:
        _deep_set(e, env_args.get("config", {}))
        e['seed'] = seed
        e['mpc']['planning_model'] = params

    # Handle errors gracefully
    total_reward = 0.0
    total_tonnes = 0.0
    try:
        while True:
            with suppress_worker_output():
                observation, reward, done, info = env.step(None)
            total_reward += reward
            total_tonnes += observation.get("total_tonnes", 0.0)

            call_count = info.get("call_count") if isinstance(info, dict) else None

            if done or (call_count is not None and call_count >= 12):
                # Get relaxation history to return
                relaxation_info = None
                if hasattr(env, 'mpc') and hasattr(env.mpc, 'relaxation_tree'):
                    history = env.mpc.relaxation_tree.get_history()
                    relaxation_info = {
                        'total_relaxations': len(history),
                        'relaxations_used': [h['name'] for h in history]
                    }
                return True, total_reward / total_tonnes if total_tonnes > 0 else 0.0, relaxation_info

    except Exception as e:
        # Get relaxation info even on failure
        relaxation_info = None
        if hasattr(env, 'mpc') and hasattr(env.mpc, 'relaxation_tree'):
            history = env.mpc.relaxation_tree.get_history()
            relaxation_info = {
                'total_relaxations': len(history),
                'relaxations_used': [h['name'] for h in history]
            }

        # Dump diagnostic files if requested
        if dump_diagnostics and hasattr(env, 'mpc'):
            try:
                env.mpc._diagnose_infeasibility()
                print(f"[Worker {id}] Diagnostic files saved for failed run", file=sys.stderr, flush=True)
            except Exception as diag_error:
                print(f"[Worker {id}] Failed to save diagnostics: {diag_error}", file=sys.stderr, flush=True)

        return False, -np.inf, relaxation_info
