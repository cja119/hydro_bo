"""
Roll out the logistics controller day by day with perfect foresight of the site
deliveries from stage 4, then check and plot the result (stage 5a).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import yaml
from pyomo.environ import value

from hydro_bo.envs.contract.dynamics import LogisticsDynamics
from hydro_bo.envs.shipping.utils import import_mpc_data, import_mpc_functions
from hydro_bo.mpc.mpc import LogisticsMPCController
from hydro_bo.utils.logging_config import configure_logging, get_logger
from hydro_bo.utils.search_space import split_env_overrides
from hydro_bo.utils.seeding import derive_subseed

logger = get_logger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[1]
LOGISTICS_DATA_DIR = ROOT_DIR / "src" / "hydro_bo" / "data" / "logistics"
LOGISTICS_EQUATIONS_PATH = ROOT_DIR / "src" / "hydro_bo" / "envs" / "contract" / "logistics_equations.py"
CONTRACT_DIR = ROOT_DIR / "paper_results" / "contract"
STAGE0_DIR = CONTRACT_DIR / "stage0"
STAGE4_DIR = CONTRACT_DIR / "stage4"
OUT_DIR = CONTRACT_DIR / "stage5a"

DESIGNS = {
    "NH3": ROOT_DIR / "paper_results" / "ablation" / "NH3" / "sobol_64_CI_NONE_comb_08"
    / "NH3-sobol_64_CI_NONE_comb_08-best.yml",
    "LH2": ROOT_DIR / "paper_results" / "ablation" / "LH2" / "sobol_64_CI_050_comb_14"
    / "LH2-sobol_64_CI_050_comb_14-best.yml",
}

SEED = 0
N_MONTHS = 11
EVALUATED_DAYS = 365
PRICE = 5.0                 # $/kg
TANK_CARGOES = 3            # inventory_upper in cargoes of the largest ship
SETTINGS = {
    "commitment": 0,
    "half_spread": 0,
    "inventory_lower": 0,
    "std_ship_arrival_time": 0,
}
PLOT_MONTH = 3
TOL = 1e-6

SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
LIMIT_COLOR = "0.35"


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def build_logistics(design, delivered, settings=None, seed=SEED):
    params, _ = split_env_overrides(design)
    controller_data = import_mpc_data(
        dict(params),
        design["vector"],
        data_folder=LOGISTICS_DATA_DIR,
    )
    controller_data.update(
        import_mpc_functions(
            LOGISTICS_DATA_DIR,
            controller_data["sets"],
            equations_path=LOGISTICS_EQUATIONS_PATH,
        )
    )

    p = controller_data["params"]
    largest_cargo = max(p["ship_capacity"].values()) * p["calorific_value"] / 120 / 1000  # kt H2
    p.update(SETTINGS)
    p["inventory_upper"] = TANK_CARGOES * largest_cargo
    p.update(settings or {})

    controller = LogisticsMPCController(gurobi_seed=derive_subseed(seed, "gurobi"))
    controller.build(controller_data)
    dynamics = LogisticsDynamics(controller_data, {"seed": seed}, delivered, PRICE)
    return controller, dynamics, controller_data


def record_day(inst, dynamics, day, month, solve_s, status, relaxations):
    ships = list(inst.ships)
    upper = getattr(inst, "cons2")[24]
    storage_end = value(inst.vector_storage[24])
    states = dynamics.ship_states(day)
    return {
        "day": day,
        "month": month,
        "delivered_t": value(inst.vector_delivered[24]),
        "storage_start": value(inst.vector_storage[0]),
        "storage_end": storage_end,
        "storage_limit": storage_end - value(upper.body),
        "flared": value(inst.vector_flared[24]),
        "ordered": [round(value(inst.n_ship_ordered[s, 24])) for s in ships],
        "sent": [round(value(inst.n_ship_sent[s, 24])) for s in ships],
        "discharged": [round(value(inst.n_ship_discharged[s, 24])) for s in ships],
        "waiting_model": [value(inst.waiting_ships[s, 24]) for s in ships],
        "holding_model": [value(inst.ships_holding[s, 24]) for s in ships],
        "waiting": [states["waiting"][s] for s in ships],
        "at_sea": [states["at_sea"][s] for s in ships],
        "holding": [states["holding"][s] for s in ships],
        "buyer_inventory_start": value(inst.buyer_inventory[0]),
        "buyer_inventory": value(inst.buyer_inventory[24]),
        "spot_bought": value(inst.spot_bought[24]),
        "spot_sold": value(inst.spot_sold[24]),
        "spot_price_model": value(inst.spot_forecast[24]),                          # $/kg
        "fundamental": value(inst.fundamental_price[24]),                           # $/kg
        "spot_forecast_path": [value(inst.spot_forecast[t]) for t in inst.grid0][1:],
        "solve_s": solve_s,
        "status": status,
        "relaxations": relaxations,
    }


def run_rollout(vector, settings=None, out_path=None, seed=SEED, deliveries_dir=STAGE4_DIR):
    design_path = DESIGNS[vector]
    label = design_path.stem.removesuffix("-best")
    with open(design_path) as f:
        design = yaml.safe_load(f)
    delivered = np.load(deliveries_dir / f"{label}-site-seed_{seed}.npz")["daily_delivered_t"]

    controller, dynamics, controller_data = build_logistics(design, delivered, settings, seed)
    ships = controller_data["sets"]["ships"]
    warm_up_days = controller_data["contract"].get("warm_up_days", 30)
    n_days = warm_up_days + EVALUATED_DAYS

    rows = []
    month = 0
    while dynamics.iter_count < n_days:
        n_steps = dynamics.get_forecasts()
        month += 1
        for _ in range(n_steps):
            if dynamics.iter_count >= n_days:
                break
            day = dynamics.iter_count
            mpc_args = dynamics.get_mpc_args()
            controller.update(mpc_args, dynamics.results)

            start = time.perf_counter()
            end_states = controller.solve(solver="gurobi")
            solve_s = time.perf_counter() - start

            inst = controller.instance
            if day == 0:
                dynamics.add_initial_orders({s: round(value(inst.n_ship_ordered[s, 0])) for s in ships})

            gurobi_model = controller.solver._solver_model
            obj_val = gurobi_model.ObjVal
            bound_gap = gurobi_model.ObjBound - gurobi_model.ObjVal
            logger.info("logistics_rollout.objective", day=day, obj_val=obj_val, bound_gap=bound_gap)

            spot_price, cap_binds = dynamics.spot_price()
            dynamics.set_results(end_states)
            dynamics.simulate_logistics_dynamics()
            rows.append({
                **record_day(
                    inst,
                    dynamics,
                    day,
                    month,
                    solve_s,
                    str(controller.results.solver.termination_condition),
                    controller._relaxation_stats["relaxations_per_solve"][-1],
                ),
                "spot_price": spot_price,  # $/kg, realised
                "cap_binds": cap_binds,
                "obj_val": obj_val,        # $M
                "bound_gap": bound_gap,    # $M
                "evaluated": day >= warm_up_days,
            })
        logger.info("logistics_rollout.progress", vector=vector, month=month, days=dynamics.iter_count,
                    total_days=n_days)

    data = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    p = controller_data["params"]
    ship_records = dynamics.ships
    loading_time = p["ship_charge_limit"]
    np.savez(
        out_path or OUT_DIR / f"{label}-seed_{seed}.npz",
        label=label,
        seed=seed,
        warm_up_days=warm_up_days,
        ships=np.array(ships),
        ship_capacity=np.array([p["ship_capacity"][s] for s in ships]),
        ship_charter_rate=np.array([p["ship_charter_rate"][s] for s in ships]),
        ship_berthing_rate=np.array([p["ship_berthing_rate"][s] for s in ships]),
        mean_ship_arrival_time=p["mean_ship_arrival_time"],
        laden_transit_time=p["laden_transit_time"],
        ballast_transit_time=p["ballast_transit_time"],
        port_time=p["port_time"],
        ship_fuel_rate=np.array([p["ship_fuel_rate"][s] for s in ships]),
        port_fee=np.array([p["port_fee"][s] for s in ships]),
        holding_loss_rate=p["holding_loss_rate"],
        reconversion_efficiency=p["reconversion_efficiency"],
        reconversion_cost=p["reconversion_cost"],
        calorific_value=p["calorific_value"],
        vector_storage_capacity=p["vector_storage_capacity"],
        vector_storage_lower_backoff=p["vector_storage_lower_backoff"],
        vector_storage_upper_backoff=p["vector_storage_upper_backoff"],
        inventory_lower=p["inventory_lower"],
        inventory_upper=p["inventory_upper"],
        commitment=p["commitment"],
        half_spread=p["half_spread"],
        price=PRICE,
        loading_time=np.array([loading_time[s] if isinstance(loading_time, dict) else loading_time for s in ships]),
        arrival_time=dynamics._arrival_time,
        planned_arrival_time=p["mean_ship_arrival_time"] + p["expected_arrival_offset"],
        transit_time=dynamics._transit_time,
        ship_size=np.array([s["size"] for s in ship_records]),
        ship_ordered=np.array([s["ordered"] for s in ship_records]),
        ship_origin_arrival=np.array([-1 if s["origin_arrival"] is None else s["origin_arrival"] for s in ship_records]),
        ship_sent=np.array([-1 if s["sent"] is None else s["sent"] for s in ship_records]),
        ship_buyer_arrival=np.array([-1 if s["buyer_arrival"] is None else s["buyer_arrival"] for s in ship_records]),
        ship_discharged=np.array([-1 if s["discharged"] is None else s["discharged"] for s in ship_records]),
        **data,
    )
    logger.info("logistics_rollout.saved", vector=vector, days=len(rows))


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def monthly_comparison(d, label):
    """Vector sent [kt] and ships sent by size per month: logistics, integrated, site deliveries."""
    cap_kt = d["ship_capacity"] / 1000
    stage0 = np.load(STAGE0_DIR / f"{label}-seed_{SEED}.npz")
    stage4 = np.load(STAGE4_DIR / f"{label}-site-seed_{SEED}.npz")
    rows = []
    for m in range(1, N_MONTHS + 1):
        sent = d["sent"][d["month"] == m].sum(axis=0)
        sent_integrated = stage0["n_ship_sent"][stage0["month"] == m].sum(axis=0)
        rows.append({
            "month": m,
            "logistics_kt": float(sent @ cap_kt),
            "logistics_ships": sent,
            "integrated_kt": float(sent_integrated @ (stage0["ship_capacity"] / 1000)),
            "integrated_ships": sent_integrated,
            "site_kt": float(stage4["daily_delivered_t"][stage4["day_month"] == m].sum() / 1000),
        })
    return rows


def report(vector):
    label = DESIGNS[vector].stem.removesuffix("-best")
    d = np.load(OUT_DIR / f"{label}-seed_{SEED}.npz")
    ships = [str(s) for s in d["ships"]]
    cap_kt = d["ship_capacity"] / 1000
    sent_kt = d["sent"] @ cap_kt

    print(f"\n=== {vector}  ({label}, seed {SEED}, {len(d['day'])} days, "
          f"arrival {int(d['arrival_time'])} d (planned {int(d['planned_arrival_time'])} d), "
          f"transit {int(d['transit_time'])} d, tank 0-{float(d['inventory_upper']):.1f} kt H2) ===")
    print("   " + ", ".join(f"{s}: storage_capacity_s {c:.0f} t, loading_time {lt:g}"
                         for s, c, lt in zip(ships, d["ship_capacity"], d["loading_time"])))

    # 1. Origin mass balance
    residual = d["delivered_t"] / 1000 - (sent_kt + d["storage_end"] - d["storage_start"] + d["flared"])
    carry = d["storage_start"][1:] - d["storage_end"][:-1]
    cumulative = (d["delivered_t"].sum() / 1000
                  - (sent_kt.sum() + d["storage_end"][-1] - d["storage_start"][0] + d["flared"].sum()))
    print("\n1. Origin mass balance  delivered = sent + d(storage) + burnt  [kt]")
    print(f"   max daily residual {np.abs(residual).max():.3e}, cumulative residual {cumulative:.3e}, "
          f"max day-to-day storage carry-over gap {np.abs(carry).max():.3e}")
    print(f"   totals: delivered {d['delivered_t'].sum() / 1000:.1f}, sent {sent_kt.sum():.1f}, "
          f"d(storage) {d['storage_end'][-1] - d['storage_start'][0]:+.1f}, burnt {d['flared'].sum():.1f}")

    # 2. Ship conservation
    cum_sent = np.cumsum(d["sent"], axis=0)
    cum_discharged = np.cumsum(d["discharged"], axis=0)
    ship_residual = cum_sent - (cum_discharged + d["at_sea"] + d["holding"])
    print("\n2. Ship conservation  sent = discharged + at sea + d(holding)  [ships, cumulative]")
    print(f"   max residual {np.abs(ship_residual).max()}; "
          f"model vs tracked: waiting {np.abs(d['waiting_model'] - d['waiting']).max():.1e}, "
          f"holding {np.abs(d['holding_model'] - d['holding']).max():.1e}")
    print("   " + ", ".join(f"{s}: sent {int(cum_sent[-1, j])}, discharged {int(cum_discharged[-1, j])}, "
                         f"at sea {int(d['at_sea'][-1, j])}, holding {int(d['holding'][-1, j])}"
                         for j, s in enumerate(ships)))

    # 3. Tank band
    lower, upper = float(d["inventory_lower"]), float(d["inventory_upper"])
    outside = (d["buyer_inventory"] < lower - TOL) | (d["buyer_inventory"] > upper + TOL)
    print(f"\n3. Buyer tank within [{lower:.1f}, {upper:.1f}] kt H2 every day: {not outside.any()} "
          f"(min {d['buyer_inventory'].min():.2f}, max {d['buyer_inventory'].max():.2f}, "
          f"days outside {int(outside.sum())})")

    # 4. Burning days
    burning = np.flatnonzero(d["flared"] > TOL)
    print(f"\n4. Days with burning: {len(burning)}  (storage at end of day / limit in force that day)")
    for i in burning:
        print(f"   day {int(d['day'][i]):>3} (month {int(d['month'][i]):>2}): burnt {d['flared'][i]:7.2f} kt, "
              f"storage {d['storage_end'][i]:7.2f} / {d['storage_limit'][i]:7.2f} kt "
              f"= {d['storage_end'][i] / d['storage_limit'][i]:.3f}")

    # 5. Holding
    print(f"\n5. Ship-days holding at the buyer: {int(d['holding'].sum())} "
          f"(model {d['holding_model'].sum():.2f})")

    # 6. Monthly comparison
    print("\n6. Monthly vector sent [kt vector] and ships sent "
          f"({'/'.join(ships)}): logistics | integrated (stage 0) | site deliveries (stage 4)")
    for r in monthly_comparison(d, label):
        print(f"   month {r['month']:>2}: logistics {r['logistics_kt']:7.1f} "
              f"[{'/'.join(str(int(n)) for n in r['logistics_ships'])}] | "
              f"integrated {r['integrated_kt']:7.1f} [{'/'.join(str(int(n)) for n in r['integrated_ships'])}] | "
              f"site {r['site_kt']:7.1f}")

    # 7. Weekly orders
    weekly = np.add.reduceat(d["ordered"].sum(axis=1), np.arange(0, len(d["day"]), 7))
    print(f"\n7. Ships ordered per week ({len(weekly)} weeks, last partial):")
    print("   " + " ".join(str(int(n)) for n in weekly))

    # 8. Solve times
    not_optimal = int((d["status"] != "optimal").sum())
    print(f"\n8. Solves: {len(d['solve_s'])}, mean {d['solve_s'].mean():.3f} s, max {d['solve_s'].max():.3f} s, "
          f"not optimal {not_optimal} ({sorted(set(d['status']) - {'optimal'}) or 'none'}), "
          f"with relaxation {int((d['relaxations'] > 0).sum())}")

    return d, label


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_storage(d, vector):
    m = d["month"] == PLOT_MONTH
    days = d["day"][m] + 1
    cap = float(d["vector_storage_capacity"])
    fig, ax = plt.subplots(1, 1, figsize=(9.0, 3.8))
    ax.step(days, d["storage_end"][m], where="mid", color=SERIES_COLORS[0], linewidth=1.6,
            label="Origin storage (end of day)")
    ax.plot(days, d["storage_limit"][m], color=LIMIT_COLOR, linewidth=0.9, label="Limit in force")
    for frac, lbl in ((float(d["vector_storage_upper_backoff"]), "Backoff limits (t ≥ forecast horizon)"),
                      (float(d["vector_storage_lower_backoff"]), None)):
        ax.axhline(cap * frac, color=LIMIT_COLOR, linestyle="--", linewidth=0.8, alpha=0.8, label=lbl)
    burn = d["flared"][m] > TOL
    ax.scatter(days[burn], d["storage_end"][m][burn], s=40, marker="v", color=SERIES_COLORS[1],
               edgecolors="white", linewidths=0.8, zorder=4, label="Burning")
    ax.set_title(f"Origin Storage — {vector}, month {PLOT_MONTH}", fontsize=9, loc="left", pad=2)
    ax.set_xlabel("Day", fontsize=9)
    ax.set_ylabel("Vector storage [kt]", fontsize=9)
    ax.set_ylim(bottom=0)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7, frameon=False)
    return _save(fig, f"{vector}-origin_storage_month{PLOT_MONTH}.png")


def plot_gantt(d, vector):
    m = d["month"] == PLOT_MONTH
    first, last = int(d["day"][m][0]), int(d["day"][m][-1]) + 1
    end = len(d["day"])

    def until(x):
        return end if x < 0 else x

    phases = [
        ("Ordered", d["ship_ordered"], np.array([until(x) for x in d["ship_origin_arrival"]])),
        ("Waiting", d["ship_origin_arrival"], np.array([-1 if a < 0 else until(x) for a, x in zip(d["ship_origin_arrival"], d["ship_sent"])])),
        ("At sea", d["ship_sent"], np.array([-1 if s < 0 else b for s, b in zip(d["ship_sent"], d["ship_buyer_arrival"])])),
        ("Holding", d["ship_buyer_arrival"], np.array([until(x) for x in d["ship_discharged"]])),
        ("Discharging", d["ship_discharged"], np.array([-1 if x < 0 else x + 1 for x in d["ship_discharged"]])),
    ]
    active = [
        i for i in range(len(d["ship_size"]))
        if d["ship_ordered"][i] < last and until(d["ship_discharged"][i]) + 1 > first
    ]

    fig, ax = plt.subplots(1, 1, figsize=(10.0, 0.32 * len(active) + 1.4))
    for row, i in enumerate(active):
        for color, (name, starts, ends) in zip(SERIES_COLORS, phases):
            s, e = starts[i], ends[i]
            if s < 0 or e < 0 or e <= s:
                continue
            ax.barh(row, e - s, left=s, height=0.6, color=color, edgecolor="white", linewidth=1.0,
                    label=name if row == 0 or name not in ax.get_legend_handles_labels()[1] else None)
    ax.set_yticks(range(len(active)))
    ax.set_yticklabels([f"{d['ship_size'][i]} #{i}" for i in active], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(first, last)
    ax.set_title(f"Ships — {vector}, month {PLOT_MONTH}", fontsize=9, loc="left", pad=2)
    ax.set_xlabel("Day", fontsize=9)
    ax.grid(True, axis="x", linewidth=0.4, alpha=0.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7, frameon=False)
    return _save(fig, f"{vector}-ships_gantt_month{PLOT_MONTH}.png")


def plot_monthly(results):
    fig, axes = plt.subplots(1, len(results), figsize=(5.5 * len(results), 3.8), squeeze=False)
    width = 0.27
    for ax, (vector, (d, label)) in zip(axes[0], results.items()):
        rows = monthly_comparison(d, label)
        months = np.array([r["month"] for r in rows])
        for k, (key, name) in enumerate((("logistics_kt", "Logistics layer"),
                                         ("integrated_kt", "Integrated controller"),
                                         ("site_kt", "Site deliveries"))):
            ax.bar(months + (k - 1) * width, [r[key] for r in rows], width=width,
                   color=SERIES_COLORS[k], edgecolor="white", linewidth=1.0, label=name)
        ax.set_title(f"Monthly Vector Sent — {vector}", fontsize=9, loc="left", pad=2)
        ax.set_xlabel("Env month", fontsize=9)
        ax.set_ylabel("Vector [kt/month]", fontsize=9)
        ax.set_xticks(months)
        ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)
        ax.legend(loc="lower right", fontsize=7, frameon=False)
    return _save(fig, "monthly_vector_sent.png")


def plot_tank(d, vector):
    days = d["day"] + 1
    fig, ax = plt.subplots(1, 1, figsize=(9.0, 3.8))
    ax.step(days, d["buyer_inventory"], where="mid", color=SERIES_COLORS[0], linewidth=1.4,
            label="Buyer inventory (end of day)")
    ax.axhline(float(d["inventory_upper"]), color=LIMIT_COLOR, linewidth=0.9, label="Band")
    ax.axhline(float(d["inventory_lower"]), color=LIMIT_COLOR, linewidth=0.9)
    ax.set_title(f"Buyer Tank — {vector}", fontsize=9, loc="left", pad=2)
    ax.set_xlabel("Day", fontsize=9)
    ax.set_ylabel("Inventory [kt H2]", fontsize=9)
    ax.set_xlim(days[0], days[-1])
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7, frameon=False)
    return _save(fig, f"{vector}-buyer_tank.png")


def _save(fig, name):
    fig.tight_layout()
    path = OUT_DIR / name
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_logging(log_file=OUT_DIR / "logistics_rollout.log")

    if not args.report_only:
        for vector in args.vectors:
            run_rollout(vector)

    plt.rcParams["font.family"] = "serif"
    results = {}
    paths = []
    for vector in args.vectors:
        d, label = report(vector)
        results[vector] = (d, label)
        paths += [plot_storage(d, vector), plot_gantt(d, vector), plot_tank(d, vector)]
    paths.append(plot_monthly(results))
    print()
    for path in paths:
        print(f"Saved: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Logistics rollout with perfect foresight of site deliveries.")
    parser.add_argument("--vectors", type=str, nargs="+", default=sorted(DESIGNS), choices=sorted(DESIGNS),
                        help="Carriers to run (default: both).")
    parser.add_argument("--report_only", action="store_true",
                        help="Skip the rollouts; check and plot the saved .npz files.")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
