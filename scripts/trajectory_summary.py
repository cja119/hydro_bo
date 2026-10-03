"""
Summarise the hourly trajectories written by `trajectory_log.py`.

Loads every `<label>-seed_<n>.npz` in `paper_results/contract/stage0/`,
groups them by design label, and prints per design (seeds pooled):
  - fraction of hours with vector storage within 2% / 5% of the upper
    effective limit (|x - U| <= tol * U), plus the fraction above U
  - fraction of hours with n_active_trains_conversion below its maximum,
    binned by vector storage as a fraction of U
  - monthly sent volume [t H2-eq]: mean, median, std, lag-1 autocorrelation

Plots per design: histogram of monthly sent volume (seeds pooled) and a
box plot of monthly sent volume by month.

Months are env months (one `env.step` call each); they are 28-31 days
long, and because each seed starts the weather data at a random offset,
the same month index is not the same season across seeds.
"""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STAGE0_DIR = Path(__file__).resolve().parents[1] / "paper_results" / "contract" / "stage0"

UPPER_TOLS = [0.02, 0.05]
# Vector storage / upper effective limit. The last bin catches hours above
# the limit, which the MPC allows (limits only bind for t >= forecast horizon).
STORAGE_BINS = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0), (1.0, np.inf)]

SERIES_COLOR = "#2a78d6"


def load_runs():
    runs = defaultdict(list)
    for path in sorted(STAGE0_DIR.glob("*-seed_*.npz")):
        d = dict(np.load(path))
        if "monthly_sent_t_h2" not in d:
            print(f"Skipping {path.name}: written by an older trajectory_log.py")
            continue
        runs[str(d["label"])].append(d)
    return runs


def lag1_autocorr(series_list):
    """Lag-1 autocorrelation pooled over seeds: deviations from the pooled
    mean, with lag pairs taken only within each seed's series."""
    mean = np.concatenate(series_list).mean()
    num = sum(np.sum((x[:-1] - mean) * (x[1:] - mean)) for x in series_list)
    den = sum(np.sum((x - mean) ** 2) for x in series_list)
    return float(num / den) if den > 0 else float("nan")


def summarise(label, runs):
    seeds = [int(d["seed"]) for d in runs]
    vs = np.concatenate([d["vector_storage"] for d in runs])
    upper = np.concatenate([d["vector_storage_upper"] for d in runs])
    trains = np.concatenate([d["n_active_trains_conversion"] for d in runs])
    max_trains = np.concatenate([np.full(len(d["hour"]), d["conversion_trains_number"]) for d in runs])
    ratio = vs / upper

    print(f"\n=== {label}  (seeds {seeds}, {len(vs)} hours) ===")
    print(f"Upper effective limit U = {float(runs[0]['vector_storage_upper'][0]):.2f} kt "
          f"(capacity {float(runs[0]['vector_storage_capacity']):.2f} kt "
          f"x backoff {float(runs[0]['vector_storage_upper_backoff']):.3f})")

    print("\nVector storage near the upper effective limit")
    for tol in UPPER_TOLS:
        frac = np.mean(np.abs(vs - upper) <= tol * upper)
        print(f"  within {tol:.0%} of U : {frac:6.1%}")
    print(f"  above U          : {np.mean(vs > upper):6.1%}")

    print("\nConversion trains below maximum, by vector storage / U")
    print(f"  {'bin':>10}  {'hours':>6}  {'frac below max':>14}")
    for lo, hi in STORAGE_BINS:
        in_bin = (ratio >= lo) & ((ratio <= hi) if hi == 1.0 else (ratio < hi))
        name = f">{lo:.0%}" if np.isinf(hi) else f"{lo:.0%}-{hi:.0%}"
        n = int(in_bin.sum())
        frac = np.mean(trains[in_bin] < max_trains[in_bin]) if n else float("nan")
        print(f"  {name:>10}  {n:>6}  {frac:>14.1%}")

    monthly = [d["monthly_sent_t_h2"] for d in runs]
    pooled = np.concatenate(monthly)
    print("\nMonthly sent volume [t H2-eq]")
    print(f"  {'':>8}  {'mean':>9}  {'median':>9}  {'std':>9}  {'lag-1 r':>8}")
    for seed, x in zip(seeds, monthly):
        print(f"  {f'seed {seed}':>8}  {x.mean():>9.1f}  {np.median(x):>9.1f}  "
              f"{x.std(ddof=1):>9.1f}  {lag1_autocorr([x]):>8.3f}")
    print(f"  {'pooled':>8}  {pooled.mean():>9.1f}  {np.median(pooled):>9.1f}  "
          f"{pooled.std(ddof=1):>9.1f}  {lag1_autocorr(monthly):>8.3f}")


def plot_monthly(label, runs):
    plt.rcParams["font.family"] = "serif"
    monthly = np.stack([d["monthly_sent_t_h2"] for d in runs])  # (n_seeds, 12)
    months = runs[0]["months"]
    days = np.round(runs[0]["monthly_hours"] / 24).astype(int)

    # --- Histogram, seeds pooled ---
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 3.8))
    ax.hist(monthly.ravel(), bins="auto", color=SERIES_COLOR, edgecolor="white", linewidth=1.0)
    ax.set_title(f"Monthly Sent Volume — {label} ({len(runs)} seeds pooled)", fontsize=9, loc="left", pad=2)
    ax.set_xlabel("Sent volume per month [t H2-eq]", fontsize=9)
    ax.set_ylabel("Months [count]", fontsize=9)
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    path = STAGE0_DIR / f"{label}-monthly_sent_hist.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved: {path}")

    # --- Box plot by month, seeds overlaid as points ---
    fig, ax = plt.subplots(1, 1, figsize=(9.0, 3.8))
    ax.boxplot(
        [monthly[:, i] for i in range(monthly.shape[1])],
        positions=months, widths=0.6, showfliers=False,
        medianprops={"color": SERIES_COLOR, "linewidth": 1.5},
        boxprops={"linewidth": 0.8}, whiskerprops={"linewidth": 0.8}, capprops={"linewidth": 0.8},
    )
    for i, d in enumerate(runs):
        jitter = (i - (len(runs) - 1) / 2) * 0.12
        ax.scatter(months + jitter, monthly[i], s=16, color=SERIES_COLOR, alpha=0.7,
                   edgecolors="white", linewidths=0.6, zorder=3,
                   label="Seed values" if i == 0 else None)
    ax.set_xticks(months)
    ax.set_xticklabels([f"{m}\n({n} d)" for m, n in zip(months, days)], fontsize=8)
    ax.set_title(f"Monthly Sent Volume by Month — {label} (n={len(runs)} per month)",
                 fontsize=9, loc="left", pad=2)
    ax.set_xlabel("Env month", fontsize=9)
    ax.set_ylabel("Sent volume [t H2-eq]", fontsize=9)
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)
    ax.legend(loc="best", fontsize=7, frameon=False)
    fig.tight_layout()
    path = STAGE0_DIR / f"{label}-monthly_sent_box.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    runs = load_runs()
    if not runs:
        print(f"No trajectory .npz files found in {STAGE0_DIR}")
        return
    for label, design_runs in runs.items():
        summarise(label, design_runs)
        plot_monthly(label, design_runs)


if __name__ == "__main__":
    main()
