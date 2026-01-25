from sys import argv

from hydro_bo.envs import Planning


def run_planning(vector_type: str):
    environment = Planning(
        f"{vector_type}-Chile", weather_file="CoastalChile_15-20_Wind.csv"
    )

    with environment as env:
        env["booleans"]["vector_choice"][vector_type] = True
        env["booleans"]["electrolysers"]["SOFC"] = True
        env["booleans"]["wind"] = True
        env["equipment"]["vector_production"][vector_type] = (
            4 if vector_type == "NH3" else 12
        )

    environment.solve()
    environment.get_results()


if __name__ == "__main__":
    assert argv[1] in ["NH3", "LH2"], "Please provide a valid vector type: NH3 or LH2"

    print(f"Planning completed for vector type: {argv[1]}")
    run_planning(argv[1])
    print(f"Planning completed, results saved to src/tmp/planning/{argv[1]}-Chile.yml")
