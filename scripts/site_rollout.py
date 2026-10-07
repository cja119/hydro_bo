"""
Roll out the site production controller for the stage-0 designs.

Runs `SiteMPCController` on the site model (`data/site/`, `site_equations.py`)
daily for `N_MONTHS` months per seed, with the same weather, seeding and
warm-start pattern as `ShippingEnv`, and saves hourly production and daily
delivered vector to `paper_results/contract/stage4/`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import yaml
import numpy as np
from pyomo.environ import value

from hydro_bo.envs.contract.dynamics import SiteDynamics
from hydro_bo.envs.shipping.utils import args_dict, import_mpc_data, import_mpc_functions
from hydro_bo.mpc.mpc import SiteMPCController
from hydro_bo.utils.logging_config import configure_logging, get_logger
from hydro_bo.utils.run_config import load_config, merge_env_overrides
from hydro_bo.utils.search_space import split_env_overrides
from hydro_bo.utils.seeding import derive_subseed

logger = get_logger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPTS_DIR.parent
SITE_DATA_DIR = ROOT_DIR / "src" / "hydro_bo" / "data" / "site"
SITE_EQUATIONS_PATH = ROOT_DIR / "src" / "hydro_bo" / "envs" / "contract" / "site_equations.py"
OUT_DIR = ROOT_DIR / "paper_results" / "contract" / "stage4"

# Stage-0 designs.
DESIGNS = {
    "NH3": ROOT_DIR / "paper_results" / "ablation" / "NH3" / "sobol_64_CI_NONE_comb_08"
    / "NH3-sobol_64_CI_NONE_comb_08-best.yml",
    "LH2": ROOT_DIR / "paper_results" / "ablation" / "LH2" / "sobol_64_CI_050_comb_14"
    / "LH2-sobol_64_CI_050_comb_14-best.yml",
}

SEEDS = [0, 1, 2]
N_MONTHS = 12
WINDOW_HOURS = 24
HOURLY_VARS = [
    "vector_flux",
    "n_active_trains_conversion",
    "energy_curtailed",
    "hydrogen_storage",
]


def _deep_set(target, updates):
    for k, v in updates.items():
        if isinstance(v, dict):
            if k not in target or not isinstance(target[k], dict):
                target[k] = {}
            _deep_set(target[k], v)
        else:
            target[k] = v


def build_site(design, seed):
    """Site controller and dynamics, set up as `ShippingEnv._build_env` does."""
    cfg = load_config(SCRIPTS_DIR / "config.yml", vector_override=design["vector"])
    params, env_overrides = split_env_overrides(design)
    env_args = merge_env_overrides(cfg, env_overrides)

    args = args_dict()
    _deep_set(args, env_args["config"])
    args["seed"] = seed

    controller = SiteMPCController(gurobi_seed=derive_subseed(seed, "gurobi"))
    controller_data = import_mpc_data(
        dict(params),
        args["vector"],
        args["mpc"]["random_param"],
        random_param_seed=derive_subseed(seed, "random_param"),
        data_folder=SITE_DATA_DIR,
    )
    controller_data.update(
        import_mpc_functions(
            SITE_DATA_DIR,
            controller_data["sets"],
            equations_path=SITE_EQUATIONS_PATH,
        )
    )
    controller.build(controller_data)
    dynamics = SiteDynamics(controller_data, args)
    return controller, dynamics


def run_rollout(design, seed):
    controller, dynamics = build_site(design, seed)

    rows = []
    daily_delivered = []
    month_of_day = []
    for month in range(1, N_MONTHS + 1):
        n_steps = dynamics.get_forecasts()
        for _ in range(n_steps):
            controller.update(dynamics.get_mpc_args(), dynamics.results)
            end_states, delivered, _ = controller.solve(solver="gurobi")
            dynamics.set_results(end_states)

            # Window 0 keeps t=0; later windows start at t=1 (t=0 is the previous t=24)
            inst = controller.instance
            window = dynamics.iter_count
            for t in range(0 if window == 0 else 1, WINDOW_HOURS + 1):
                row = {"hour": window * WINDOW_HOURS + t, "window": window, "month": month}
                for name in HOURLY_VARS:
                    row[name] = float(value(getattr(inst, name)[t]))
                rows.append(row)
            daily_delivered.append(delivered)
            month_of_day.append(month)

            dynamics.simulate_site_dynamics()

        logger.info("site_rollout.progress", seed=seed, month=month, total=N_MONTHS,
                    windows=dynamics.iter_count)

    data = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    data["daily_delivered_t"] = np.array(daily_delivered)
    data["day_month"] = np.array(month_of_day)
    return data


def main(args):
    design_path = DESIGNS[args.vector]
    label = design_path.stem.removesuffix("-best")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seeds_tag = "_".join(str(s) for s in args.seeds)
    configure_logging(log_file=OUT_DIR / f"{label}-site-seeds_{seeds_tag}.log")

    with open(design_path) as f:
        design = yaml.safe_load(f)
    logger.info("site_rollout.start", design=str(design_path), seeds=args.seeds, months=N_MONTHS)

    for seed in args.seeds:
        data = run_rollout(design, seed)
        path = OUT_DIR / f"{label}-site-seed_{seed}.npz"
        np.savez(
            path,
            seed=seed,
            label=label,
            design_path=str(design_path.relative_to(ROOT_DIR)),
            **data,
        )
        logger.info("site_rollout.npz_saved", path=str(path), n_hours=len(data["hour"]),
                    annual_delivered_t=float(data["daily_delivered_t"].sum()))

    logger.info("site_rollout.complete", out_dir=str(OUT_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roll out the site production controller.")
    parser.add_argument("--vector", type=str, required=True, choices=sorted(DESIGNS),
                        help="Carrier whose stage-0 design to run.")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                        help=f"Env master seeds, one rollout each (default: {SEEDS}).")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
