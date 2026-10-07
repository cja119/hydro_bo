"""
One month of the integrated controller or the site layer for a stage-0 design, seed 0,
with the MIP gap overridden for every controller.

Produces paper_results/contract/stage4/mipgap_0.001/ with
    python scripts/mipgap_month.py <integrated|site> <NH3|LH2> 0.001 \
        paper_results/contract/stage4/mipgap_0.001/<integrated|site>
"""

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hydro_bo.mpc.mpc import MPCController
from hydro_bo.utils.logging_config import get_logger

logger = get_logger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parent
N_MONTHS = 1
SEED = 0


def override_mip_gap(mip_gap):
    """Set the MIP gap on every controller's solver after its usual configuration."""
    ensure_solver = MPCController._ensure_solver

    def ensure_solver_with_gap(self, solver):
        configured = self._solver_configured
        ensure_solver(self, solver)
        if not configured:
            self.solver.options["mipgap"] = mip_gap
            logger.info("mipgap_month.solver_options", mipgap=self.solver.options["mipgap"],
                        threads=self.solver.options["Threads"],
                        time_limit=self.solver.options["TimeLimit"])

    MPCController._ensure_solver = ensure_solver_with_gap


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(mode, vector, mip_gap, out_dir):
    override_mip_gap(mip_gap)
    site_rollout = load_script("site_rollout")

    if mode == "integrated":
        sys.argv = ["trajectory_log.py", "--design", str(site_rollout.DESIGNS[vector]),
                    "--seeds", str(SEED)]
        trajectory_log = load_script("trajectory_log")
        trajectory_log.N_MONTHS = N_MONTHS
        trajectory_log.OUT_DIR = out_dir
        trajectory_log.main(trajectory_log.parse_args())
    else:
        sys.argv = ["site_rollout.py", "--vector", vector, "--seeds", str(SEED)]
        site_rollout.N_MONTHS = N_MONTHS
        site_rollout.OUT_DIR = out_dir
        site_rollout.main(site_rollout.parse_args())


if __name__ == "__main__":
    assert len(sys.argv) == 5, "Usage: python mipgap_month.py <integrated|site> <NH3|LH2> <mip_gap> <out_dir>"
    assert sys.argv[1] in ["integrated", "site"], "Mode must be integrated or site"
    assert sys.argv[2] in ["NH3", "LH2"], "Vector must be NH3 or LH2"

    main(sys.argv[1], sys.argv[2], float(sys.argv[3]), Path(sys.argv[4]))
