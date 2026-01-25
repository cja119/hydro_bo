from hydro_bo.algs import BayesianOptimizer
import sys
import os


def run_planning(vector_type: str):

    # Optimizer setup
    optimizer = BayesianOptimizer("../src/hydro_bo/tmp")
    with optimizer as config:
        config["vector_type"] = vector_type
        config["planning_results_file"] = f"{vector_type}-Chile.yml"
        config["scale"] = (0.5, 1.5)
        config["use_planning_bounds"] = True

    # Run optimization
    results = optimizer.optimize(
        num_samples=int(sys.argv[2]),
        experiment_name=f"{vector_type}_bayesopt",
        n_cores=8,
    )

    return optimizer.get_best_result()


if __name__ == "__main__":
    assert sys.argv[1] in [
        "NH3",
        "LH2",
    ], "Please provide a valid vector type: NH3 or LH2"

    print("Running Bayesian Optimization for vector type:", sys.argv[1])
    best = run_planning(sys.argv[1])
    print("Best configuration found:", best)
