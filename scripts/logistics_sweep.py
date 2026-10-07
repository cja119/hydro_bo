"""
Sweep the logistics rollout over the contract commitment, buyer tank size and spot
half-spread (stage 5c), then report the spot trades and imbalance cost of each run
against the monthly tank post-processing of Request 4, applied both to the Part A
monthly volumes and to the stage-0 monthly volumes (seed SEED, months 1..N_MONTHS).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
from itertools import product

import matplotlib.pyplot as plt
import numpy as np

import contract_postprocess
import logistics_rollout
from logistics_rollout import DESIGNS, N_MONTHS, PRICE, SEED, run_rollout
from hydro_bo.utils.logging_config import configure_logging, get_logger

logger = get_logger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[1]
PART_A_DIR = ROOT_DIR / "paper_results" / "contract" / "stage5a"
OUT_DIR = ROOT_DIR / "paper_results" / "contract" / "stage5c"

CARGO_OFFSETS = [-1.0, -0.5, 0.0, 0.5, 1.0]   # X = median monthly delivery + offset x large cargo
TANK_CARGOES = [1.5, 2.0, 3.0]                # inventory_upper = n x large cargo
EPSILONS = [0.0, 0.05, 0.1, 0.2]
Z = 2.0
SIGMA_T_DAYS = 0.66
DAYS_PER_MONTH = 30.0
PLOT_N = 2.0
PLOT_EPS = 0.1
TOL = 1e-4                                    # kt H2, tank at a limit

N_COLORS = {1.5: "#2a78d6", 2.0: "#eb6834", 3.0: "#1baf7a"}
N_MARKERS = {1.5: "o", 2.0: "s", 3.0: "^"}
LIMIT_COLOR = "0.35"


def label_of(vector):
    return DESIGNS[vector].stem.removesuffix("-best")


def cargo_h2(d):
    """Hydrogen equivalent of one cargo of each ship size [kt H2]."""
    return d["ship_capacity"] * float(d["calorific_value"]) / 120 / 1000


def part_a(vector):
    """Median monthly hydrogen discharged in Part A and the largest cargo [kt H2]."""
    d = np.load(PART_A_DIR / f"{label_of(vector)}-seed_{SEED}.npz")
    discharged = d["discharged"] @ cargo_h2(d)
    monthly = [discharged[d["month"] == m].sum() for m in range(1, N_MONTHS + 1)]
    return float(np.median(monthly)), float(cargo_h2(d).max())


def run_path(vector, offset, n, eps):
    return OUT_DIR / f"{label_of(vector)}-X{offset:+g}-n{n:g}-eps{eps:g}.npz"


def grid(vector, arrival_offset=None):
    median, large = part_a(vector)
    for offset, n, eps in product(CARGO_OFFSETS, TANK_CARGOES, EPSILONS):
        commitment = median + offset * large
        settings = {
            "commitment": commitment,
            "half_spread": eps,
            "inventory_upper": n * large,
            "inventory_lower": Z * SIGMA_T_DAYS * commitment / DAYS_PER_MONTH,
        }
        if arrival_offset is not None:
            settings["expected_arrival_offset"] = arrival_offset
        yield offset, n, eps, settings


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metrics(d):
    eps, commitment = float(d["half_spread"]), float(d["commitment"])
    sold, bought = d["spot_sold"], d["spot_bought"]
    discharged = d["discharged"] @ cargo_h2(d)
    revenue = (
        PRICE * commitment / DAYS_PER_MONTH
        + (1 - eps) * PRICE * sold
        - (1 + eps) * PRICE * bought
    )  # $M per day
    tank_change = float(d["buyer_inventory"][-1] - d["buyer_inventory_start"][0])
    per_year = 365 / len(d["day"])
    imbalance = eps * PRICE * (sold.sum() + bought.sum())
    burnt_h2 = float(d["flared"].sum()) * float(d["calorific_value"]) / 120
    orders = np.array([(d["ship_size"] == s).sum() for s in d["ships"]])
    charter = (
        orders @ (float(d["order_fee"]) + d["ship_charter_rate"] * float(d["charter_duration"]) * 24)
        + d["waiting"].sum(axis=0) @ ((d["ship_berthing_rate"] + d["ship_charter_rate"]) * 24)
        + d["holding"].sum(axis=0) @ (d["ship_charter_rate"] * 24)
    ) / 1_000_000
    return {
        "sold": sold.sum(),
        "bought": bought.sum(),
        "residual": revenue.sum() - PRICE * discharged.sum() + imbalance + PRICE * tank_change,
        "imbalance_per_year": imbalance * per_year,
        "burnt_h2": burnt_h2,
        "burn_loss_per_year": PRICE * burnt_h2 * per_year,
        "charter_per_year": charter * per_year,
        "total_per_year": (imbalance + PRICE * burnt_h2 + charter) * per_year,
        "revenue_per_year": (revenue.sum() + PRICE * tank_change) * per_year,
        "holding": int(d["holding"].sum()),
        "burnt": float(d["flared"].sum()),
        "days_lower": int((d["buyer_inventory"] <= float(d["inventory_lower"]) + TOL).sum()),
        "days_upper": int((d["buyer_inventory"] >= float(d["inventory_upper"]) - TOL).sum()),
        "ships": d["sent"].sum(axis=0),
        "tank_change": tank_change,
    }


def part_a_volumes(vector):
    """Part A monthly hydrogen sent [t H2], months 1..N_MONTHS, shape (1, N_MONTHS)."""
    d = np.load(PART_A_DIR / f"{label_of(vector)}-seed_{SEED}.npz")
    sent = d["sent"] @ cargo_h2(d) * 1000
    return np.array([[sent[d["month"] == m].sum() for m in range(1, N_MONTHS + 1)]])


def stage0_volumes(vector):
    """Stage-0 monthly hydrogen sent [t H2] for seed SEED, months 1..N_MONTHS, shape (1, N_MONTHS)."""
    carrier = contract_postprocess.load_carriers()[vector]
    return carrier["V"][carrier["seeds"].index(SEED)][None, :N_MONTHS]


def request4_imbalance(vector, volumes, commitment, n, eps):
    """Imbalance cost [$M/yr] of the Request 4 monthly tank (I_max = (n - 1) Q_s) on monthly volumes [t H2]."""
    carrier = contract_postprocess.load_carriers()[vector]
    X = np.array([commitment * 1000])                      # kt -> t H2
    i_min, i_max = contract_postprocess.tank_limits(n, X, carrier["Q_s"])
    sim = contract_postprocess.simulate(volumes, X, i_min, i_max)
    gross_spot = (sim["o"] + sim["u"]).mean(axis=(0, 1))   # t H2 per month
    return float(eps * contract_postprocess.PRICE * gross_spot[0] * 12 / 1e6)


# ---------------------------------------------------------------------------
# Report and plots
# ---------------------------------------------------------------------------

def report(vector):
    median, large = part_a(vector)
    rows = {}
    for offset, n, eps, settings in grid(vector):
        path = run_path(vector, offset, n, eps)
        if path.exists():
            rows[offset, n, eps] = {**metrics(np.load(path)), "X": settings["commitment"]}

    print(f"\n=== {vector}  ({label_of(vector)}, seed {SEED}; median monthly H2 {median:.2f} kt, "
          f"large cargo {large:.2f} kt H2; {len(rows)} runs) ===")
    print("  imbalance cost [$M/yr]: (i) this run, (ii) Request 4 on Part A volumes, "
          "(iii) Request 4 on stage-0 volumes")
    print("  burnt in kt vector and kt H2; burn loss, charter and total = (i) + burn loss + charter in $M/yr")
    print(f"  {'X':>6} {'n':>4} {'eps':>5} {'sold o':>8} {'bought u':>8} {'residual':>9} "
          f"{'(i)':>7} {'(ii)':>7} {'(iii)':>7} {'hold':>5} {'burnt':>6} {'burnt H2':>8} {'burn $M':>8} "
          f"{'charter':>8} {'total':>8} {'@lower':>6} {'@upper':>6}  ships S/M/L")
    part_a_v, stage0_v = part_a_volumes(vector), stage0_volumes(vector)
    for offset, n in product(CARGO_OFFSETS, TANK_CARGOES):
        X = median + offset * large
        for eps in EPSILONS:
            r = rows.get((offset, n, eps))
            if r is None:
                continue
            r["r4_part_a_per_year"] = request4_imbalance(vector, part_a_v, X, n, eps)
            r["r4_stage0_per_year"] = request4_imbalance(vector, stage0_v, X, n, eps)
            print(f"  {X:>6.2f} {n:>4g} {eps:>5g} {r['sold']:>8.1f} {r['bought']:>8.1f} {r['residual']:>9.1e} "
                  f"{r['imbalance_per_year']:>7.2f} {r['r4_part_a_per_year']:>7.2f} {r['r4_stage0_per_year']:>7.2f} "
                  f"{r['holding']:>5} {r['burnt']:>6.1f} {r['burnt_h2']:>8.2f} {r['burn_loss_per_year']:>8.2f} "
                  f"{r['charter_per_year']:>8.2f} {r['total_per_year']:>8.2f} {r['days_lower']:>6} {r['days_upper']:>6}  "
                  f"{'/'.join(str(int(k)) for k in r['ships'])}")
    return rows


def plot_imbalance(all_rows):
    fig, axes = plt.subplots(1, len(all_rows), figsize=(5.5 * len(all_rows), 3.8), squeeze=False)
    for ax, (vector, rows) in zip(axes[0], all_rows.items()):
        for n in TANK_CARGOES:
            this = [rows[0.0, n, eps]["imbalance_per_year"] for eps in EPSILONS]
            part_a = [rows[0.0, n, eps]["r4_part_a_per_year"] for eps in EPSILONS]
            stage0 = [rows[0.0, n, eps]["r4_stage0_per_year"] for eps in EPSILONS]
            ax.plot(EPSILONS, this, color=N_COLORS[n], marker=N_MARKERS[n], markersize=6, linewidth=1.6,
                    label=f"n={n:g}")
            ax.plot(EPSILONS, part_a, color=N_COLORS[n], linewidth=1.2, linestyle=":")
            ax.plot(EPSILONS, stage0, color=N_COLORS[n], linewidth=1.2, linestyle="--", alpha=0.8)
        ax.plot([], [], color=LIMIT_COLOR, linestyle="-", label="This run (solid)")
        ax.plot([], [], color=LIMIT_COLOR, linestyle=":", label="Request 4, Part A volumes (dotted)")
        ax.plot([], [], color=LIMIT_COLOR, linestyle="--", label="Request 4, stage-0 volumes (dashed)")
        ax.set_title(f"Imbalance Cost at Median X — {vector}", fontsize=9, loc="left", pad=2)
        ax.set_xlabel("Half-spread ε [-]", fontsize=9)
        ax.set_ylabel("Imbalance cost [$M/yr]", fontsize=9)
        ax.set_xticks(EPSILONS)
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.legend(loc="upper left", fontsize=7, frameon=False)
    return _save(fig, "imbalance_vs_eps.png")


def plot_holding(all_rows):
    fig, axes = plt.subplots(1, len(all_rows), figsize=(5.5 * len(all_rows), 3.8), squeeze=False)
    for ax, (vector, rows) in zip(axes[0], all_rows.items()):
        for n in TANK_CARGOES:
            ax.plot(EPSILONS, [rows[0.0, n, eps]["holding"] for eps in EPSILONS], color=N_COLORS[n],
                    marker=N_MARKERS[n], markersize=6, linewidth=1.6, label=f"n={n:g}")
        ax.set_title(f"Ship-Days Holding at Median X — {vector}", fontsize=9, loc="left", pad=2)
        ax.set_xlabel("Half-spread ε [-]", fontsize=9)
        ax.set_ylabel("Ship-days holding", fontsize=9)
        ax.set_xticks(EPSILONS)
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.legend(loc="upper left", fontsize=7, frameon=False)
    return _save(fig, "holding_vs_eps.png")


def plot_revenue(all_rows):
    fig, axes = plt.subplots(1, len(all_rows), figsize=(5.5 * len(all_rows), 3.8), squeeze=False)
    for ax, (vector, rows) in zip(axes[0], all_rows.items()):
        for n in TANK_CARGOES:
            X = [rows[o, n, PLOT_EPS]["X"] for o in CARGO_OFFSETS]
            ax.plot(X, [rows[o, n, PLOT_EPS]["revenue_per_year"] for o in CARGO_OFFSETS], color=N_COLORS[n],
                    marker=N_MARKERS[n], markersize=6, linewidth=1.6, label=f"n={n:g}")
        ax.set_title(f"Revenue at ε={PLOT_EPS:g} (tank change valued at P) — {vector}",
                     fontsize=9, loc="left", pad=2)
        ax.set_xlabel("Commitment X [kt H2/month]", fontsize=9)
        ax.set_ylabel("Revenue [$M/yr]", fontsize=9)
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.legend(loc="lower left", fontsize=7, frameon=False)
    return _save(fig, "revenue_vs_X.png")


def best_offset(rows):
    return max(CARGO_OFFSETS, key=lambda o: rows[o, PLOT_N, PLOT_EPS]["revenue_per_year"])


def plot_tank_month(d, vector):
    month = logistics_rollout.PLOT_MONTH
    m = d["month"] == month
    days = d["day"][m] + 1
    fig, ax = plt.subplots(1, 1, figsize=(9.0, 3.8))
    ax.step(days, d["buyer_inventory"][m], where="mid", color=N_COLORS[1.5], linewidth=1.6,
            label="Buyer inventory (end of day)")
    ax.axhline(float(d["inventory_upper"]), color=LIMIT_COLOR, linewidth=0.9, label="Band")
    ax.axhline(float(d["inventory_lower"]), color=LIMIT_COLOR, linewidth=0.9)
    level = d["buyer_inventory"][m]
    for mask, marker, color, name in (
        (d["spot_sold"][m] > TOL, "v", N_COLORS[2.0], "Spot sale"),
        (d["spot_bought"][m] > TOL, "^", N_COLORS[3.0], "Spot purchase"),
        (d["discharged"][m].sum(axis=1) > 0, "D", "#eda100", "Discharge"),
    ):
        ax.scatter(days[mask], level[mask], s=40, marker=marker, color=color, edgecolors="white",
                   linewidths=0.8, zorder=4, label=name)
    ax.set_title(f"Buyer Tank — {vector}, month {month}, n={PLOT_N:g}, ε={PLOT_EPS:g}, "
                 f"X={float(d['commitment']):.1f} kt/month", fontsize=9, loc="left", pad=2)
    ax.set_xlabel("Day", fontsize=9)
    ax.set_ylabel("Inventory [kt H2]", fontsize=9)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7, frameon=False)
    return _save(fig, f"{vector}-buyer_tank_month{month}.png")


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
    global OUT_DIR
    if args.out_dir is not None:
        OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.report_only:
        configure_logging(log_file=OUT_DIR / f"sweep-{args.vector}-worker{args.worker}.log")
        runs = list(grid(args.vector, args.arrival_offset))[args.worker::args.n_workers]
        for offset, n, eps, settings in runs:
            path = run_path(args.vector, offset, n, eps)
            if path.exists():
                continue
            run_rollout(args.vector, settings=settings, out_path=path)
            logger.info("logistics_sweep.run_saved", path=str(path))
        return

    plt.rcParams["font.family"] = "serif"
    all_rows = {vector: report(vector) for vector in sorted(DESIGNS)}
    paths = [plot_imbalance(all_rows), plot_holding(all_rows), plot_revenue(all_rows)]

    logistics_rollout.OUT_DIR = OUT_DIR
    for vector, rows in all_rows.items():
        offset = best_offset(rows)
        d = np.load(run_path(vector, offset, PLOT_N, PLOT_EPS))
        print(f"\n{vector}: best X at n={PLOT_N:g}, eps={PLOT_EPS:g} is {float(d['commitment']):.2f} kt H2/month "
              f"(median {offset:+g} large cargo)")
        paths += [plot_tank_month(d, vector), logistics_rollout.plot_gantt(d, vector)]
    print()
    for path in paths:
        print(f"Saved: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Logistics sweep over commitment, tank size and half-spread.")
    parser.add_argument("--vector", type=str, choices=sorted(DESIGNS), help="Carrier to run.")
    parser.add_argument("--worker", type=int, default=0, help="Index of this worker.")
    parser.add_argument("--n_workers", type=int, default=1, help="Number of workers splitting the grid.")
    parser.add_argument("--report_only", action="store_true", help="Report and plot the saved runs.")
    parser.add_argument("--arrival_offset", type=int, default=None,
                        help="expected_arrival_offset for every run (default: the design's).")
    parser.add_argument("--out_dir", type=str, default=None, help=f"Output directory (default: {OUT_DIR}).")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
