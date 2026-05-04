"""XLA thread configuration for scripts that drive JAX-backed code.

The CPU parallelism story has two knobs:

  - `n_devices`: number of logical CPU devices XLA exposes. `pmap`
    shards work over these. Each device runs one Cholesky / cho_solve
    call sequentially.
  - `blas_threads_per_device`: BLAS threads each device's Cholesky uses.

Total CPU saturation = `n_devices × blas_threads_per_device`. For the
acquisition Sobol screen (16 384 candidates → 16 384 Cholesky-solves),
many independent devices win over fatter BLAS — `n_devices = num_cores,
blas_threads = 1` is usually the right shape on CPU.

Must be called BEFORE the first `import jax` anywhere in the process —
XLA reads its flags out of the environment at first init and won't
pick up later changes.
"""

import os
import sys

from hydro_bo.utils.logging_config import get_logger

logger = get_logger(__name__)


def configure_jax_threads(n_devices: int, blas_threads_per_device: int = 1) -> None:
    """Configure XLA logical-device count + BLAS threads per device.

    Sets the relevant env vars (`XLA_FLAGS`, `OMP_NUM_THREADS`, etc.)
    so that:
      - `jax.local_device_count()` returns `n_devices` (CPU only).
      - Each device's Cholesky uses `blas_threads_per_device` threads.

    No-op (with warning) if `jax` has already been imported."""
    if "jax" in sys.modules:
        logger.warning(
            "configure_jax_threads_late",
            message="jax already imported — XLA flags will not take effect",
            n_devices=int(n_devices),
            blas_threads_per_device=int(blas_threads_per_device),
        )
        return

    nd = max(1, int(n_devices))
    bt = max(1, int(blas_threads_per_device))

    os.environ.setdefault("JAX_ENABLE_X64", "1")

    flags = [
        f"--xla_force_host_platform_device_count={nd}",
        "--xla_cpu_multi_thread_eigen=true",
        f"intra_op_parallelism_threads={bt}",
        f"inter_op_parallelism_threads={bt}",
    ]
    existing = os.environ.get("XLA_FLAGS", "")
    os.environ["XLA_FLAGS"] = (existing + " " + " ".join(flags)).strip()

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(bt)
