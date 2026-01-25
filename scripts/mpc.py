from sys import argv

from hydro_bo.envs import ShippingEnv


def run_shipping(vector_type: str):
    environment = ShippingEnv("shipping-plot")

    with environment as env:
        env["vector"] = vector_type
        env["mpc"]["planning_model"] = f"{vector_type}-Chile.yml"
        env["weather_data"]["weather_file"] = "CoastalChile_15-20_Wind.csv"

    # Advance the environment through 12 steps (months)
    for _ in range(12):
        for _ in environment.step(None):
            pass

    return environment.plot_data()


if __name__ == "__main__":
    assert argv[1] in ["NH3", "LH2"], "Vector must be NH3 or LH2"

    data = run_shipping(argv[1])
    summary = list(data) if hasattr(data, "__iter__") else type(data)

    print(f"Shipping MPC run complete for {argv[1]} (plot data keys: {summary})")
