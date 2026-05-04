"""XLA thread configuration for scripts that drive JAX-backed code."""

import os
import sys

from hydro_bo.utils.logging_config import get_logger

logger = get_logger(__name__)


def configure_jax_threads(n_threads: int) -> None:
    """Set XLA / BLAS thread counts. No-op (with warning) if jax is
    already imported — flags will not take effect at that point."""
    if "jax" in sys.modules:
        logger.warning(
            "configure_jax_threads_late",
            message="jax already imported — XLA flags will not take effect",
            n_threads=int(n_threads),
        )
        return
    n = max(1, int(n_threads))
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    flags = [
        "--xla_cpu_multi_thread_eigen=true",
        f"intra_op_parallelism_threads={n}",
        f"inter_op_parallelism_threads={n}",
    ]
    existing = os.environ.get("XLA_FLAGS", "")
    os.environ["XLA_FLAGS"] = (existing + " " + " ".join(flags)).strip()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(n)
