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

from algs.logging_config import get_logger

logger = get_logger(__name__)

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
    def __init__(self, env_args, param_overrides, num_instances, num_devices, timeout, exit_fraction):
        self._env_args = env_args
        self._params = param_overrides
        self._num_instances = num_instances
        self._timeout = timeout
        self._exit_fraction = exit_fraction

        if ray.is_initialized():
            ray.shutdown()
        ray.init(num_cpus=num_devices, runtime_env={"env_vars": {"PYTHONPATH": _PROJECT_ROOT}})

    def run_multisim(self):
        import time as _time

        start_time = _time.perf_counter()
        scores = []
        n_success = 0

        while n_success < self._exit_fraction * self._num_instances:
            elapsed = _time.perf_counter() - start_time
            if elapsed >= self._timeout:
                break
            n_required = int(self._exit_fraction * self._num_instances) - n_success
            tasks = [run_mpc.remote(self._env_args, id=f"iter_{i}", params=self._params) for i in range(n_required)]
            results = ray.get(tasks)
            n_success += sum(1 for success, _ in results if success)
            scores.extend([score for success, score in results if success])
            logger.info("multisim_progress", n_success=n_success, n_total=self._num_instances, elapsed_seconds=_time.perf_counter() - start_time)

        return scores

@ray.remote
def run_mpc(env_args, id, params):
    import structlog
    from hydro_bo import ShippingEnv

    # Silence structlog in worker — it writes directly to stderr via PrintLogger
    # so logging.setLevel() has no effect; redirect the factory to /dev/null.
    _devnull = open(os.devnull, "w")
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(file=_devnull))

    # Derive an integer seed from the Ray task ID hex string
    ctx = ray.get_runtime_context()
    seed = int(ctx.get_task_id(), 16) % (2**31)

    def _deep_set(target, updates):
        for k, v in updates.items():
            if isinstance(v, dict):
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
                return True, total_reward / total_tonnes if total_tonnes > 0 else 0.0

    except Exception as e:
        logger.error("mpc_run_failed", instance_id=id, error=str(e))
        return False, -np.inf
