"""
Smoke test for RayMultiMPC — verifies Ray initialises and parameters
can be injected from a planning model solve into a ShippingEnv.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from hydro_bo import configure_logging
from src.algs.dispatcher import RayMultiMPC

configure_logging()

PLANNING_MODEL = "NH3-Chile.yml"
VECTOR = "NH3"
WEATHER_FILE = "CoastalChile_15-20_Wind.csv"
PLANNING_MODEL_PATH = Path(__file__).parent.parent / "src/tmp/planning" / PLANNING_MODEL


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
    print(f"[INFO] Loaded planning model: {PLANNING_MODEL_PATH}")
    print(f"[INFO] Parameters: {params}")

    dispatcher = RayMultiMPC(
        env_args=ENV_ARGS,
        param_overrides=params,
        num_instances=48,
        num_devices=8,
        timeout=900,
        exit_fraction=1.0,
    )

    print("[INFO] RayMultiMPC initialised successfully.")
    print(f"[INFO] num_instances={dispatcher._num_instances}, timeout={dispatcher._timeout}")

    scores = dispatcher.run_multisim()
    print(f"[INFO] run_multisim complete. Scores: {scores}")

    import matplotlib.pyplot as plt

    plt.hist(scores, bins=10)
    plt.title("Distribution of MPC Scores")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.savefig(Path(__file__).parent.parent / "src/tmp/multi_mpc_scores.png")
    print(f"[INFO] Score distribution plot saved to src/tmp/multi_mpc_scores.png")


if __name__ == "__main__":
    main()
