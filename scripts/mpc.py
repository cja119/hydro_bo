from sys import argv
from pathlib import Path
import sys

from hydro_bo import ShippingEnv
from hydro_bo import configure_logging

# Configure structured logging
configure_logging()

PLOT_DIR = Path(__file__).parent.parent / "src/tmp/shipping_plots"

def run_shipping(vector_type: str):
    environment = ShippingEnv("shipping-plot")

    with environment as env:
        env["vector"] = vector_type
        env["mpc"]["planning_model"] = f"{vector_type}-Chile.yml"
        env["weather_data"]["weather_file"] = "CoastalChile_15-20_Wind.csv"

    # Advance the environment until 12 months are reached
    while True:
        for _ in environment.step(None):
            pass

        info = environment.last_transition[3] 
        call_count = info.get("call_count")

        print(f"[INFO] Shipping MPC - Month {call_count}/12")
        if call_count >= 12:
            break

    fig_sched, fig_energy = environment.plot_figures()
    fig_sched.savefig(PLOT_DIR / f"{vector_type}_shipping_schedule.png")
    fig_energy.savefig(PLOT_DIR / f"{vector_type}_shipping_energy.png")


if __name__ == "__main__":
    assert argv[1] in ["NH3", "LH2"], "Vector must be NH3 or LH2"

    print(f"[INFO] Running Shipping MPC for vector type: {argv[1]}. Note that if the month count resets, this is fine just the solver re-initializing.")
    run_shipping(argv[1])
    print(f"[INFO] Shipping MPC run complete for {argv[1]} (plots saved to src/tmp/shipping_plots/{argv[1]}_shipping_*.png)")