"""
Roll out ContractEnv (daily site then logistics solves) for the stage-0 designs and
compare it with the perfect-foresight logistics runs of Request 12j (stage 6).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse

import matplotlib.pyplot as plt
import numpy as np
import yaml

import logistics_sweep as ls
from hydro_bo import ShippingEnv
from hydro_bo.utils.logging_config import configure_logging, get_logger
from hydro_bo.utils.run_config import load_config, merge_env_overrides
from hydro_bo.utils.search_space import split_env_overrides

logger = get_logger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parent
CONTRACT_DIR = SCRIPTS_DIR.parent / "paper_results" / "contract"
STAGE4_DIR = CONTRACT_DIR / "stage4"
PERFECT_DIR = CONTRACT_DIR / "stage5j"
OUT_DIR = CONTRACT_DIR / "stage6"

SEED = 0
CLIMATOLOGY_SEEDS = [0, 1, 2]  # stage-4 seeds for the monthly delivery means, excluding the seed run
N_MONTHS = 11
PRICE = 5.0                  # $/kg
EPSILONS = [0.05, 0.2]
COMMITMENT = {"LH2": 57.40, "NH3": 52.54}  # kt H2 landed per 30 days
TANK_CARGOES = 1.5                         # inventory_upper in cargoes of the largest ship
FORECAST_DAYS = [1, 7, 30]
PLOT_MONTH = 3
TOL = 1e-6

SERIES_COLORS = ["#2a78d6", "#eb6834"]
LIMIT_COLOR = "0.35"


def label_of(vector):
    return ls.label_of(vector)


def monthly_delivery(vector, seed=SEED, deliveries_dir=STAGE4_DIR):
    """Mean daily delivery per env month [t vector] over the stage-4 climatology seeds other than `seed`."""
    daily, months = [], []
    for other in CLIMATOLOGY_SEEDS:
        if other == seed:
            continue
        d = np.load(deliveries_dir / f"{label_of(vector)}-site-seed_{other}.npz")
        daily.append(d["daily_delivered_t"])
        months.append((d["day_month"] - 1) % 12 + 1)
    daily, months = np.concatenate(daily), np.concatenate(months)
    return [float(daily[months == m].mean()) for m in range(1, 13)]


def settings_of(vector, eps, factor=1.0):
    _, large = ls.part_a(vector)
    commitment = factor * COMMITMENT[vector]
    return {
        "commitment": commitment,
        "half_spread": eps,
        "inventory_upper": TANK_CARGOES * large,
        "inventory_lower": ls.Z * ls.SIGMA_T_DAYS * commitment / ls.DAYS_PER_MONTH,
        "expected_arrival_offset": 0,
        "std_ship_arrival_time": 0,
    }


def _deep_set(target, updates):
    for k, v in updates.items():
        if isinstance(v, dict):
            if k not in target or not isinstance(target[k], dict):
                target[k] = {}
            _deep_set(target[k], v)
        else:
            target[k] = v


def run_path(vector, eps, seed=SEED):
    return OUT_DIR / f"{label_of(vector)}-eps{eps:g}-seed_{seed}.npz"


def run_contract(vector, eps, seed=SEED, deliveries_dir=STAGE4_DIR, premium=0.0, factor=1.0, depth=1.0):
    with open(ls.DESIGNS[vector]) as f:
        design = yaml.safe_load(f)
    cfg = load_config(SCRIPTS_DIR / "config.yml", vector_override=design["vector"])
    params, env_overrides = split_env_overrides(design)
    env_args = merge_env_overrides(cfg, env_overrides)

    env = ShippingEnv("contract")
    with env as e:
        _deep_set(e, env_args["config"])
        e["seed"] = seed
        e["mpc"]["planning_model"] = dict(params)
        settings = settings_of(vector, eps, factor)
        e["contract"]["logistics_settings"] = {**settings, "reference_volume": depth * COMMITMENT[vector] / 30}
        e["contract"]["monthly_delivery"] = monthly_delivery(vector, seed, deliveries_dir)
        e["contract"]["price"] = PRICE
        e["contract"]["price_settings"] = {"premium": premium}

    done = False
    while not done:
        observation, reward, done, info = env.step(None)
        logger.info("contract_rollout.progress", vector=vector, eps=eps, seed=seed, month=info["call_count"],
                    month_cash_flow=reward)

    records = observation["records"]
    data = {k: np.array([r[k] for r in records]) for k in records[0]}
    p = env._logistics_data["params"]
    ships = list(env._logistics_data["sets"]["ships"])
    ship_records = env._dynamics.ships
    np.savez(
        run_path(vector, eps, seed),
        label=label_of(vector),
        seed=seed,
        warm_up_days=env._warm_up_days,
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
        export_jetty_capex=p["export_jetty_capex"],
        n_berths=p["n_berths"],
        calorific_value=p["calorific_value"],
        vector_storage_capacity=p["vector_storage_capacity"],
        inventory_lower=p["inventory_lower"],
        inventory_upper=p["inventory_upper"],
        commitment=p["commitment"],
        half_spread=p["half_spread"],
        reference_volume=p["reference_volume"],
        penalty=env._logistics_data["price_dynamics"].get("penalty", 9.0),
        premium=premium,
        contract_factor=factor,
        **{k: p[k] for k in p if k.endswith("_backoff")},
        price=PRICE,
        ship_size=np.array([s["size"] for s in ship_records]),
        ship_ordered=np.array([s["ordered"] for s in ship_records]),
        ship_origin_arrival=np.array([-1 if s["origin_arrival"] is None else s["origin_arrival"] for s in ship_records]),
        ship_sent=np.array([-1 if s["sent"] is None else s["sent"] for s in ship_records]),
        ship_buyer_arrival=np.array([-1 if s["buyer_arrival"] is None else s["buyer_arrival"] for s in ship_records]),
        ship_discharged=np.array([-1 if s["discharged"] is None else s["discharged"] for s in ship_records]),
        **data,
    )
    logger.info("contract_rollout.saved", path=str(run_path(vector, eps, seed)))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def cargo_h2(d):
    return d["ship_capacity"] * float(d["calorific_value"]) / 120 / 1000


def storage_price(d):
    """Origin storage value per kg H2: fundamental price less large-ship charter, as the objective."""
    j = [str(s) for s in d["ships"]].index("large")
    charter = d["ship_charter_rate"][j] * float(d["charter_duration"]) * 24 / 1_000_000
    return float(d["price"]) - charter / cargo_h2(d)[j]


def daily_charter(d):
    """Charter cost per day [$M], with the day -1 orders charged on day 0."""
    order_cost = (float(d["order_fee"]) + d["ship_charter_rate"] * float(d["charter_duration"]) * 24) / 1_000_000
    ordered = d["ordered"].astype(float).copy()
    ordered[0] += [((d["ship_size"] == s) & (d["ship_ordered"] == -1)).sum() for s in d["ships"]]
    return (
        ordered @ order_cost
        + d["waiting"] @ ((d["ship_berthing_rate"] + d["ship_charter_rate"]) * 24 / 1_000_000)
        + d["holding"] @ (d["ship_charter_rate"] * 24 / 1_000_000)
    )


def stock_value(d):
    """Value of origin storage, ships sent but not discharged, and the buyer's tank at the end of each day [$M]."""
    price = float(d["price"])
    sizes = [str(s) for s in d["ships"]]
    days = d["day"]
    afloat = np.zeros(len(days))
    for size, sent, discharged in zip(d["ship_size"], d["ship_sent"], d["ship_discharged"]):
        if sent < 0:
            continue
        end = len(days) if discharged < 0 else discharged
        afloat[(days >= sent) & (days < end)] += cargo_h2(d)[sizes.index(str(size))]
    storage_h2 = d["storage_end"] * float(d["calorific_value"]) / 120
    return storage_price(d) * storage_h2 + price * afloat + price * d["buyer_inventory"]


def monthly_net_value(d):
    """Spot revenue less spot purchases and charter, plus the change in stock value, per env month [$M]."""
    price, eps = float(d["price"]), float(d["half_spread"])
    cash = (1 - eps) * price * d["spot_sold"] - (1 + eps) * price * d["spot_bought"] - daily_charter(d)
    stock = stock_value(d)
    start = storage_price(d) * d["storage_start"][0] * float(d["calorific_value"]) / 120 + price * d["buyer_inventory_start"][0]
    out, previous = [], start
    for m in range(1, N_MONTHS + 1):
        last = np.flatnonzero(d["month"] == m)[-1]
        out.append(cash[d["month"] == m].sum() + stock[last] - previous)
        previous = stock[last]
    return np.array(out)


def forecast_error(d, k):
    """Mean absolute error of the day-k delivery forecast against the delivery that day [t vector]."""
    actual = d["delivered_t"]
    n = len(actual) - k
    return float(np.mean(np.abs(d["delivery_forecast"][:n, k] - actual[k:])))


def report(vectors):
    print(f"  {'run':<22} {'net value':>9} {'imbal':>7} {'charter':>8} {'burn':>5} | {'ships S/M/L':>11} | "
          f"{'site mean/max s':>15} {'logistics mean/max s':>20} {'t-lim':>5}")
    for vector in vectors:
        for eps in EPSILONS:
            c = np.load(run_path(vector, eps))
            pf = np.load(PERFECT_DIR / f"run_{vector}_eps{eps}.npz")
            print(f"{vector} eps={eps:g}")
            for name, d in (("12j perfect foresight", pf), ("13: site + logistics", c)):
                m = ls.metrics(d)
                site = f"{d['site_solve_s'].mean():.2f} / {d['site_solve_s'].max():.1f}" if "site_solve_s" in d.files else "n/a"
                limited = int((d["status"] != "optimal").sum())
                if "site_status" in d.files:
                    limited = f"{limited}+{int((d['site_status'] != 'optimal').sum())}"
                print(f"  {name:<22} {monthly_net_value(d).sum():>9.1f} {m['imbalance_per_year']:>7.2f} {m['charter_per_year']:>8.2f} "
                      f"{m['burn_loss_per_year']:>5.2f} | {'/'.join(str(int(n)) for n in d['sent'].sum(axis=0)):>11} | "
                      f"{site:>15} {d['solve_s'].mean():>10.2f} / {d['solve_s'].max():<7.1f} {limited:>5}")

            stage4 = np.load(STAGE4_DIR / f"{label_of(vector)}-site-seed_{SEED}.npz")["daily_delivered_t"][:len(c["day"])]
            sent_kt = c["sent"] @ (c["ship_capacity"] / 1000)
            residual = c["delivered_t"] / 1000 - (sent_kt + c["storage_end"] - c["storage_start"] + c["flared"])
            lower, upper = float(c["inventory_lower"]), float(c["inventory_upper"])
            outside = (c["buyer_inventory"] < lower - TOL) | (c["buyer_inventory"] > upper + TOL)
            errors = ", ".join(f"day {k}: {forecast_error(c, k):.0f} t ({forecast_error(c, k) / c['delivered_t'].mean():.0%})"
                               for k in FORECAST_DAYS)
            print(f"    forecast MAE {errors} (mean delivery {c['delivered_t'].mean():.0f} t/day)")
            print(f"    origin mass balance max residual {np.abs(residual).max():.1e} kt; logistics day-0 delivery = site actual: "
                  f"{np.allclose(c['delivered_t'], c['site_delivered_t'])}; site deliveries = stage-4 seed {SEED}: "
                  f"{np.allclose(c['site_delivered_t'], stage4)}; tank in band every day: {not outside.any()} "
                  f"({c['buyer_inventory'].min():.2f}-{c['buyer_inventory'].max():.2f} of {lower:.2f}-{upper:.2f} kt H2)")


def plot_storage(vectors):
    fig, axes = plt.subplots(1, len(vectors), figsize=(5.5 * len(vectors), 3.8), squeeze=False)
    for ax, vector in zip(axes[0], vectors):
        d = np.load(run_path(vector, EPSILONS[0]))
        m = np.flatnonzero(d["month"] == PLOT_MONTH)
        first = m[0]
        days = d["day"][m] + 1
        plan = d["storage_plan"][first][1:len(m) + 1]
        ax.step(days, d["storage_end"][m], where="mid", color=SERIES_COLORS[0], linewidth=1.6, label="Actual")
        ax.plot(days, plan, color=SERIES_COLORS[1], linestyle="--", linewidth=1.4,
                label=f"Plan of day {int(d['day'][first])}")
        ax.axhline(float(d["vector_storage_capacity"]), color=LIMIT_COLOR, linewidth=0.8, label="Capacity")
        ax.set_title(f"Origin Storage — {vector}, month {PLOT_MONTH}, ε={EPSILONS[0]:g}", fontsize=9, loc="left", pad=2)
        ax.set_xlabel("Day", fontsize=9)
        ax.set_ylabel("Vector storage [kt]", fontsize=9)
        ax.set_ylim(bottom=0)
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.legend(loc="lower left", fontsize=7, frameon=False)
    return _save(fig, "origin_storage_plan_vs_actual.png")


def plot_monthly_net_value(vectors):
    fig, axes = plt.subplots(1, len(vectors), figsize=(5.5 * len(vectors), 3.8), squeeze=False)
    months = np.arange(1, N_MONTHS + 1)
    for ax, vector in zip(axes[0], vectors):
        for color, eps in zip(SERIES_COLORS, EPSILONS):
            c = monthly_net_value(np.load(run_path(vector, eps)))
            pf = monthly_net_value(np.load(PERFECT_DIR / f"run_{vector}_eps{eps}.npz"))
            ax.plot(months, c, color=color, marker="o", markersize=5, linewidth=1.6, label=f"ε={eps:g}, site + logistics")
            ax.plot(months, pf, color=color, marker="s", markersize=4, linewidth=1.2, linestyle="--",
                    label=f"ε={eps:g}, perfect foresight")
        ax.axhline(0.0, color=LIMIT_COLOR, linewidth=0.6)
        ax.set_title(f"Net Value by Month — {vector}", fontsize=9, loc="left", pad=2)
        ax.set_xlabel("Env month", fontsize=9)
        ax.set_ylabel("Net value [$M]", fontsize=9)
        ax.set_xticks(months)
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.legend(loc="best", fontsize=7, frameon=False)
    return _save(fig, "net_value_by_month.png")


def _save(fig, name):
    fig.tight_layout()
    path = OUT_DIR / name
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        configure_logging(log_file=OUT_DIR / f"{label_of(args.vector)}-eps{args.eps:g}-seed_{args.seed}.log")
        run_contract(args.vector, args.eps, args.seed, Path(args.deliveries_dir), args.premium, args.contract_factor,
                     args.depth)
        return

    plt.rcParams["font.family"] = "serif"
    vectors = sorted(ls.DESIGNS)
    report(vectors)
    print()
    for path in (plot_storage(vectors), plot_monthly_net_value(vectors)):
        print(f"Saved: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roll out ContractEnv and compare with perfect foresight.")
    parser.add_argument("--vector", type=str, choices=sorted(ls.DESIGNS), help="Carrier to run.")
    parser.add_argument("--eps", type=float, choices=EPSILONS, help="Half-spread to run.")
    parser.add_argument("--seed", type=int, default=SEED, help=f"Env master seed (default: {SEED}).")
    parser.add_argument("--deliveries_dir", type=str, default=str(STAGE4_DIR),
                        help="Folder of the site rollouts used for the climatology.")
    parser.add_argument("--premium", type=float, default=0.0, help="Green premium over blue [$/kg] (default: 0).")
    parser.add_argument("--contract_factor", type=float, default=1.0, help="Contract volume as a multiple of the base X (default: 1).")
    parser.add_argument("--depth", type=float, default=1.0, help="Market depth q_ref as a multiple of the base X / 30 (default: 1).")
    parser.add_argument("--report_only", action="store_true", help="Report and plot the saved runs.")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
