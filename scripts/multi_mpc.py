"""
Smoke test for RayMultiMPC — verifies Ray initialises and parameters
can be injected from a planning model solve into a ShippingEnv.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from hydro_bo import configure_logging
from hydro_bo.algs.logging_config import get_logger
from hydro_bo.algs.dispatcher import RayMultiMPC

configure_logging()
logger = get_logger(__name__)

VECTOR = "LH2"
PLANNING_MODEL = "{vector}-Chile.yml".format(vector=VECTOR)
WEATHER_FILE = "CoastalChile_15-20_Wind.csv"
PLANNING_MODEL_PATH = Path(__file__).parent / "tmp/planning" / PLANNING_MODEL

# Set to True to dump ILP/LP files on failure and exit early
DUMP_DIAGNOSTICS_ON_FAILURE = True


def load_planning_model(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


ENV_ARGS = {
    "config": {
        "vector": VECTOR,
        "mpc": {"planning_model": PLANNING_MODEL},
        "weather_data": {"weather_file": WEATHER_FILE},
    },
}


def main():
    params = load_planning_model(PLANNING_MODEL_PATH)
    logger.info("multi_mpc.loaded_model", path=str(PLANNING_MODEL_PATH))
    logger.info("multi_mpc.parameters", params=params)

    dispatcher = RayMultiMPC(
        env_args=ENV_ARGS,
        param_overrides=params,
        num_instances=144,
        num_devices=12,
        timeout=900,
        exit_fraction=1.0,
        dump_diagnostics_on_failure=DUMP_DIAGNOSTICS_ON_FAILURE,
    )

    logger.info("multi_mpc.initialized",
                num_instances=dispatcher._num_instances,
                timeout=dispatcher._timeout)

    scores = dispatcher.run_multisim()
    logger.info("multi_mpc.simulation_complete", scores=scores)

    import matplotlib.pyplot as plt

    plt.hist(scores, bins=10)
    plt.title("Distribution of MPC Scores")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.savefig(Path(__file__).parent / "tmp/multi_mpc_scores.png")
    logger.info("multi_mpc.plot_saved", path="scripts/tmp/multi_mpc_scores.png")


if __name__ == "__main__":
    main()
