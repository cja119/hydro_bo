"""Rollouts of a design under an adoption archetype: one ContractEnv year (site MPC + logistics MPC, 30-day warm-up,
365 evaluated days) per seed, with the marginal-impact storage-netback terminal value (terminal_valuation 3) and the
contract priced at the green mid (P_b + nu). Each seed draws its own weather start, price path and ship-arrival noise; the
same seed list is used for every design (common random numbers).

`evaluate` runs many seeds of one design in parallel (one process per seed, single-threaded Gurobi) and returns the
per-seed NPVs plus diagnostics.
"""

from __future__ import annotations

import logging
import math
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from hydro_bo.archetypes.metrics import plant_costs, rollout_metrics
from hydro_bo.archetypes.scenarios import Problem

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "scripts" / "config.yml"
LOGISTICS_DATA = ROOT / "src" / "hydro_bo" / "data" / "logistics"
TANK_CARGOES = 1.5          # buyer tank upper limit, in cargoes of the largest ship
Z, SIGMA_T_DAYS, DAYS_PER_MONTH = 2.0, 0.66, 30.0   # buyer tank lower limit (arrival-time cover), as the contract studies
LEGACY_PRICE = 5.0          # ContractEnv's fixed price, used only when price dynamics are off


def _deep_set(target: dict, updates: dict) -> None:
    for k, v in updates.items():
        if isinstance(v, dict):
            if k not in target or not isinstance(target[k], dict):
                target[k] = {}
            _deep_set(target[k], v)
        else:
            target[k] = v


def logistics_params(design: dict, vector: str) -> dict:
    from hydro_bo.envs.shipping.utils import import_mpc_data
    return import_mpc_data(dict(design), vector, data_folder=LOGISTICS_DATA)["params"]


def contract_settings(design: dict, problem: Problem, lp: dict) -> dict:
    """Logistics settings for this design: contract volume, market, terminal valuation, buyer tank limits."""
    a = problem.archetype
    f = float(design["contract_factor"])
    large_fuel = float(lp["ship_fuel_rate"]["large"]) * (float(lp["laden_transit_time"]) + float(lp["ballast_transit_time"])) \
        / float(lp["ship_capacity"]["large"])
    x1 = problem.reference_output_kt30(large_fuel)              # kt H2 landed per 30 days at f = 1
    large_cargo = float(lp["ship_capacity"]["large"]) * float(lp["calorific_value"]) / 120 / 1000
    commitment = f * x1
    return {
        "commitment": commitment,
        "half_spread": a.half_spread,
        "reference_volume": a.depth_days * x1 / 30,
        "inventory_upper": TANK_CARGOES * large_cargo,
        "inventory_lower": Z * SIGMA_T_DAYS * commitment / DAYS_PER_MONTH,
        "expected_arrival_offset": 0,
        "std_ship_arrival_time": 0,
        "terminal_valuation": 3,
        "spot_daily_volume": (1 - f) * x1 / 30,
    }


def record_of(env, records: list) -> dict:
    """Stack ContractEnv records into arrays and add the logistics parameters the accounting needs."""
    data = {k: np.array([r[k] for r in records]) for k in records[0]}
    p = env._logistics_data["params"]
    ships = list(env._logistics_data["sets"]["ships"])
    sr = env._dynamics.ships
    data.update({
        "warm_up_days": env._warm_up_days, "ships": np.array(ships),
        "ship_capacity": np.array([p["ship_capacity"][s] for s in ships], dtype=float),
        "ship_charter_rate": np.array([p["ship_charter_rate"][s] for s in ships], dtype=float),
        "ship_berthing_rate": np.array([p["ship_berthing_rate"][s] for s in ships], dtype=float),
        "ship_fuel_rate": np.array([p["ship_fuel_rate"][s] for s in ships], dtype=float),
        "port_fee": np.array([p["port_fee"][s] for s in ships], dtype=float),
        "mean_ship_arrival_time": p["mean_ship_arrival_time"], "laden_transit_time": p["laden_transit_time"],
        "ballast_transit_time": p["ballast_transit_time"], "port_time": p["port_time"],
        "holding_loss_rate": p["holding_loss_rate"], "calorific_value": p["calorific_value"],
        "commitment": p["commitment"], "half_spread": p["half_spread"], "reference_volume": p["reference_volume"],
        "export_jetty_capex": p["export_jetty_capex"], "n_berths": p["n_berths"], "discount_factor": p["discount_factor"],
        "ship_size": np.array([s["size"] for s in sr]),
        "ship_sent": np.array([-1 if s["sent"] is None else s["sent"] for s in sr]),
        "ship_buyer_arrival": np.array([-1 if s["buyer_arrival"] is None else s["buyer_arrival"] for s in sr]),
        "ship_discharged": np.array([-1 if s["discharged"] is None else s["discharged"] for s in sr]),
    })
    return data


def build_env(design: dict, problem: Problem, seed: int):
    """A built ContractEnv (site + logistics MPCs constructed, nothing run) for one design, archetype and seed."""
    from hydro_bo import ShippingEnv
    from hydro_bo.utils.run_config import load_config, merge_env_overrides

    vector = problem.vector
    cfg = load_config(CONFIG_PATH, vector_override=vector)
    env_args = merge_env_overrides(cfg, {})
    planning = {k: v for k, v in design.items() if k != "contract_factor"}
    lp = logistics_params(planning, vector)
    a = problem.archetype
    env = ShippingEnv("contract")
    with env as e:
        _deep_set(e, env_args["config"])
        e["seed"] = int(seed)
        e["mpc"]["planning_model"] = planning
        e["contract"]["logistics_settings"] = contract_settings(design, problem, lp)
        e["contract"]["monthly_delivery"] = [problem.reference_delivery_t_per_day()] * 12
        e["contract"]["price"] = LEGACY_PRICE
        e["contract"]["price_settings"] = {"premium": a.premium, "penalty": a.penalty}
    return env


def run_seed(design: dict, problem: Problem, seed: int, save_npz: str | None = None) -> dict:
    """One rollout. Returns the metrics dict, or {'npv': nan, 'error': ...} if the run failed."""
    from hydro_bo.utils.logging_config import configure_logging

    t0 = time.perf_counter()
    try:
        configure_logging(level=logging.WARNING, log_file=os.devnull, tee_stderr=False, package_level=logging.WARNING)
        env = build_env(design, problem, seed)
        done = False
        while not done:
            observation, _, done, _ = env.step(None)
        d = record_of(env, observation["records"])
        if save_npz:
            np.savez(save_npz, **{k: v for k, v in d.items()})
        costs = plant_costs(design["capex"], design["opex"], float(d["export_jetty_capex"]), int(d["n_berths"]),
                            float(d["discount_factor"]))
        m = rollout_metrics(d, costs)
        m.update({"seed": int(seed), "wall_s": time.perf_counter() - t0, "error": None})
        return m
    except Exception as exc:  # a failed solve or a bad design: reported, never raised into the pool
        return {"seed": int(seed), "npv": math.nan, "wall_s": time.perf_counter() - t0,
                "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=8)}


def _run_seed_star(args):
    return run_seed(*args)


def evaluate(design: dict, problem: Problem, seeds, ncpus: int, save_dir: str | None = None) -> list[dict]:
    """Run `seeds` of one design on `ncpus` processes; results in seed order."""
    seeds = [int(s) for s in seeds]
    jobs = [(design, problem, s, None if save_dir is None else str(Path(save_dir) / f"seed_{s}.npz")) for s in seeds]
    if ncpus <= 1:
        return [run_seed(*j) for j in jobs]
    with ProcessPoolExecutor(max_workers=min(ncpus, len(jobs)), mp_context=get_context("spawn")) as pool:
        return list(pool.map(_run_seed_star, jobs))
