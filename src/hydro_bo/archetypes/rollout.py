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
import queue
import signal
import time
import traceback
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

# Flat monthly delivery forecast: the reference-design stage-4 mean daily delivery, scaled by
# the binding capacity ratio. Replaces the old 60%-of-nameplate value, which understated
# reference delivery ~22% (the reference sits at ~77-78% of nameplate). The profile is flat on
# purpose — the stage-4 monthly shape was weather-start noise, not seasonality.
REF_MEAN_DELIVERY = {"NH3": 12624.0, "LH2": 2103.0}     # t carrier/day, reference-design stage-4 mean
REF_RE_CAPACITY = {"NH3": 1825.49, "LH2": 2061.56}      # reference renewable_energy_capacity
REF_EL_CAPACITY = {"NH3": 19855.02, "LH2": 19331.56}    # reference electrolyser_capacity


class _RolloutTimeout(Exception):
    """Raised by the SIGALRM handler when one rollout exceeds the wall cap."""


def _raise_rollout_timeout(signum, frame):
    raise _RolloutTimeout()


_ROLLOUT_CAP_S = None
HARD_CAP_GRACE_S = 300          # parent SIGKILLs a worker this long after its soft cap should have fired


def rollout_cap_s() -> int:
    """Per-rollout wall cap [s] from config_archetypes.yml (evaluation.rollout_cap_s); 0 disables.
    The 300 s per-solve Gurobi TimeLimit does NOT bound a whole rollout (a pathological design has
    many solves that each hit it -> hours). This caps the single-seed rollout so such a design
    returns a failed result quickly instead, which the BO feasibility model then learns from."""
    global _ROLLOUT_CAP_S
    if _ROLLOUT_CAP_S is None:
        try:
            import yaml
            cfg = yaml.safe_load(open(ROOT / "scripts" / "config_archetypes.yml"))
            _ROLLOUT_CAP_S = int(float((cfg.get("evaluation") or {}).get("rollout_cap_s", 0)) or 0)
        except Exception:
            _ROLLOUT_CAP_S = 0
    return _ROLLOUT_CAP_S


def std_ship_arrival_time() -> float:
    """Std [days] of each ship's daily arrival-time perturbation, from the STD_SHIP_ARRIVAL_TIME env
    var (default 0.0 = deterministic arrivals). The archetype `contract_settings` has pinned this to 0
    since commit 022b6a3, overriding the 0.25 default in data/variables.yml; the env var restores it
    for an A/B run without editing code. Read from the environment (not the config file) so it is a
    per-job value the PBS script sets and spawn workers inherit, independent of --config."""
    return float(os.environ.get("STD_SHIP_ARRIVAL_TIME", 0.0) or 0.0)


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
        "reference_volume": a.q_ref_days * x1 / 30,
        "inventory_upper": TANK_CARGOES * large_cargo,
        "inventory_lower": Z * SIGMA_T_DAYS * commitment / DAYS_PER_MONTH,
        "expected_arrival_offset": 0,
        "std_ship_arrival_time": std_ship_arrival_time(),
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


def monthly_delivery_forecast(design: dict, problem: Problem) -> list[float]:
    """Flat 12-month delivery forecast [t carrier/day]: the reference-design stage-4 mean daily
    delivery scaled by the binding capacity ratio (renewables / electrolyser / conversion).
    r_conv is problem.scale — conversion capacity after the train map."""
    v = problem.vector
    r_re = float(design["renewable_energy_capacity"]) / REF_RE_CAPACITY[v]
    r_el = float(design["electrolyser_capacity"]) / REF_EL_CAPACITY[v]
    r_conv = problem.scale
    return [REF_MEAN_DELIVERY[v] * min(r_re, r_el, r_conv)] * 12


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
        # Forecast padding: planning omits wind_forecast_mean on purpose, so the site MPC pads its
        # wind forecast beyond the horizon with the data's climatological mean (not the fixed
        # reference-design wind_forecast_mean) — a deliberate archetype change.
        e["mpc"]["planning_model"] = planning
        e["contract"]["logistics_settings"] = contract_settings(design, problem, lp)
        e["contract"]["monthly_delivery"] = monthly_delivery_forecast(design, problem)
        e["contract"]["price"] = LEGACY_PRICE
        e["contract"]["price_settings"] = {"premium": a.premium, "penalty": a.penalty}
    return env


def run_seed(design: dict, problem: Problem, seed: int, save_npz: str | None = None) -> dict:
    """One rollout. Returns the metrics dict, or {'npv': nan, 'error': ...} if the run failed."""
    from hydro_bo.utils.logging_config import configure_logging

    t0 = time.perf_counter()
    cap = rollout_cap_s()
    prev_handler = None
    try:
        if cap > 0:                       # SIGALRM works: run_seed runs in the worker's main thread
            prev_handler = signal.signal(signal.SIGALRM, _raise_rollout_timeout)
            signal.alarm(cap)
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
    except _RolloutTimeout:  # exceeded the per-rollout wall cap -> a failed seed (NaN) for the BO
        return {"seed": int(seed), "npv": math.nan, "wall_s": time.perf_counter() - t0,
                "error": f"rollout_cap_{cap}s"}
    except Exception as exc:  # a failed solve or a bad design: reported, never raised into the pool
        return {"seed": int(seed), "npv": math.nan, "wall_s": time.perf_counter() - t0,
                "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=8)}
    finally:
        if prev_handler is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, prev_handler)


def _seed_worker(result_q, seed: int, args) -> None:
    """Child entry point: run one seed, put (seed, result) on the queue. Runs in the child's main
    thread, so run_seed's SIGALRM soft cap still works. A child-side crash is reported, never left
    to hang the parent (which would otherwise block on the queue forever)."""
    try:
        result_q.put((seed, run_seed(*args)))
    except BaseException as exc:
        result_q.put((seed, {"seed": int(seed), "npv": math.nan, "wall_s": 0.0,
                             "error": f"worker_crash: {type(exc).__name__}: {exc}"}))


def evaluate(design: dict, problem: Problem, seeds, ncpus: int, save_dir: str | None = None) -> list[dict]:
    """Run `seeds` of one design on up to `ncpus` worker processes; results in seed order.

    One `spawn` process per seed (single-threaded Gurobi each). `spawn`, not `fork`:
    gurobipy is not fork-safe. The conda gurobipy loads because `env.sh` resets
    PYTHONPATH (dropping the Gurobi module's 3.11 gurobipy), which the workers inherit.

    Two-level wall cap. run_seed's in-process SIGALRM (`rollout_cap_s`) handles a rollout stuck
    between solves. But SIGALRM can't break a single Gurobi optimize() stuck in C past its
    TimeLimit, and one such seed would block the whole point (the result needs every seed) and
    burn the job's walltime. So the parent here hard-SIGKILLs any worker still alive
    HARD_CAP_GRACE_S after its soft cap: that seed becomes a failed (NaN) rollout and the rest of
    the point completes normally. A dynamic scheduler keeps all `ncpus` busy as seeds finish.
    """
    seeds = [int(s) for s in seeds]
    job = {s: (design, problem, s, None if save_dir is None else str(Path(save_dir) / f"seed_{s}.npz")) for s in seeds}
    if ncpus <= 1:
        return [run_seed(*job[s]) for s in seeds]

    soft = rollout_cap_s()
    hard = soft + HARD_CAP_GRACE_S if soft > 0 else 0       # 0 -> no hard cap (relies on the children finishing)
    ctx = get_context("spawn")
    results: dict[int, dict] = {}
    running: dict[int, tuple] = {}                          # seed -> (process, monotonic start)
    todo = list(seeds)

    # A Manager queue (not a bare ctx.Queue): it lives in its own process, so SIGKILLing a stuck
    # worker cannot corrupt it — a plain Queue can be left broken if a process is killed mid-use.
    with ctx.Manager() as manager:
        result_q = manager.Queue()

        def launch(s: int) -> None:
            p = ctx.Process(target=_seed_worker, args=(result_q, s, job[s]))
            p.start()
            running[s] = (p, time.monotonic())

        for _ in range(min(ncpus, len(todo))):
            launch(todo.pop(0))
        while running:
            try:
                s, res = result_q.get(timeout=5.0)
                if s in running:                           # ignore a late put from an already-killed child
                    running.pop(s)[0].join()
                    results[s] = res
                    if todo:
                        launch(todo.pop(0))
            except queue.Empty:
                pass
            if hard:                                       # SIGKILL any rollout past the hard wall
                now = time.monotonic()
                for s, (p, t0) in list(running.items()):
                    if now - t0 > hard:
                        p.kill()
                        p.join()
                        running.pop(s, None)
                        results[s] = {"seed": s, "npv": math.nan, "wall_s": now - t0,
                                      "error": f"rollout_hardcap_{hard}s"}
                        if todo:
                            launch(todo.pop(0))
    return [results[s] for s in seeds]
