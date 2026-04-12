from sys import argv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hydro_bo import ShippingEnv
from hydro_bo import configure_logging
from src.algs.logging_config import get_logger

# Configure structured logging
configure_logging()
logger = get_logger(__name__)

PLOT_DIR = Path(__file__).parent / "tmp/shipping_plots"

def run_shipping(vector_type: str, seed: int = None):
    environment = ShippingEnv("shipping-plot")

    with environment as env:
        env["vector"] = vector_type
        env["mpc"]["planning_model"] = f"{vector_type}-Chile.yml"
        env["weather_data"]["weather_file"] = "CoastalChile_15-20_Wind.csv"
        if seed is not None:
            env["seed"] = seed

    # Advance the environment until 12 months are reached
    while True:
        for _ in environment.step(None):
            pass

        info = environment.last_transition[3]
        call_count = info.get("call_count")

        logger.info("shipping_mpc.progress", month=call_count, total=12)
        if call_count >= 12:
            break
    
    if not PLOT_DIR.exists():
        PLOT_DIR.mkdir(parents=True)
    fig_sched, fig_energy = environment.plot_figures()
    fig_sched.savefig(PLOT_DIR / f"{vector_type}_shipping_schedule.png")
    fig_energy.savefig(PLOT_DIR / f"{vector_type}_shipping_energy.png")


if __name__ == "__main__":
    assert len(argv) >= 2, "Usage: python mpc.py <NH3|LH2> [seed]"
    assert argv[1] in ["NH3", "LH2"], "Vector must be NH3 or LH2"

    seed = int(argv[2]) if len(argv) > 2 else None

    logger.info("shipping_mpc.start", vector=argv[1], seed=seed, note="Month count resets are normal during solver re-initialization")
    run_shipping(argv[1], seed=seed)
    logger.info("shipping_mpc.complete", vector=argv[1], seed=seed, plots_dir=f"scripts/tmp/shipping_plots/{argv[1]}_shipping_*.png")