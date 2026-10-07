"""
Compare site-layer rollouts (stage 4) with the integrated controller (stage 0).

For each design and seed, compares vector_flux over the applied hours 1..8760
(day d = hours 24d+1..24d+24): annual totals, the fraction of days where the
site layer is more than DAILY_TOL below the integrated controller, and the
time-limited solves and mean solve time of each controller. Prints the
report (also written to `site_compare.txt`) and saves the plots to stage4/.
"""

import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CONTRACT_DIR = Path(__file__).resolve().parents[1] / "paper_results" / "contract"
STAGE0_DIR = CONTRACT_DIR / "stage0"
STAGE4_DIR = CONTRACT_DIR / "stage4"

DAILY_TOL = 0.01
HOURS_PER_DAY = 24
PLOT_SEED = 0

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")

SITE_COLOR = "#2a78d6"
INTEGRATED_COLOR = "#eb6834"
SEED_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]


def daily_flux(d):
    """Daily vector_flux [GJ] over the applied hours 1..24*n_days."""
    flux = d["vector_flux"][1:]
    return flux.reshape(-1, HOURS_PER_DAY).sum(axis=1)


def solve_stats(log_path):
    """Time-limited solves and mean solve time [s] from a run log."""
    time_limited, solve_ms = 0, []
    for line in log_path.read_text().splitlines():
        line = ANSI_ESCAPE.sub("", line)
        if "accepting_time_limited_incumbent" in line:
            time_limited += 1
        elif "solve_complete" in line:
            solve_ms.append(float(re.search(r"solve_call_ms=([\d.]+)", line).group(1)))
    return time_limited, np.mean(solve_ms) / 1e3


def load_pairs():
    """label -> list of (seed, site daily flux, integrated daily flux, site stats, integrated stats)."""
    pairs = defaultdict(list)
    for site_path in sorted(STAGE4_DIR.glob("*-site-seed_*.npz")):
        site = np.load(site_path)
        label, seed = str(site["label"]), int(site["seed"])
        integrated = np.load(STAGE0_DIR / f"{label}-seed_{seed}.npz")
        assert np.array_equal(site["hour"], integrated["hour"]), f"{label} seed {seed}: hours differ"
        pairs[label].append((
            seed,
            daily_flux(site),
            daily_flux(integrated),
            solve_stats(STAGE4_DIR / f"{label}-site-seeds_{seed}.log"),
            solve_stats(STAGE0_DIR / f"{label}-seeds_{seed}.log"),
        ))
    return pairs


def report(label, runs):
    lines = [
        f"\n=== {label}  (seeds {[r[0] for r in runs]}, {len(runs[0][1])} days each) ===",
        f"  {'seed':>6}  {'site [TJ]':>11}  {'integrated [TJ]':>15}  {'diff':>8}  "
        f"{'days >1% below':>14}  {'time-limited site/integ':>23}  {'mean solve [s] site/integ':>25}",
    ]
    for seed, site, integrated, site_stats, integ_stats in runs:
        below = np.mean(site < (1 - DAILY_TOL) * integrated)
        limited = f"{site_stats[0]} / {integ_stats[0]}"
        solve_s = f"{site_stats[1]:.2f} / {integ_stats[1]:.2f}"
        lines.append(f"  {seed:>6}  {site.sum() / 1e3:>11.1f}  {integrated.sum() / 1e3:>15.1f}  "
                     f"{site.sum() / integrated.sum() - 1:>+8.2%}  {below:>14.1%}  "
                     f"{limited:>23}  {solve_s:>25}")
    site_all = np.concatenate([r[1] for r in runs])
    integ_all = np.concatenate([r[2] for r in runs])
    lines.append(f"  {'pooled':>6}  {site_all.sum() / 1e3:>11.1f}  {integ_all.sum() / 1e3:>15.1f}  "
                 f"{site_all.sum() / integ_all.sum() - 1:>+8.2%}  "
                 f"{np.mean(site_all < (1 - DAILY_TOL) * integ_all):>14.1%}")
    return lines


def plot_daily(pairs):
    fig, axes = plt.subplots(len(pairs), 1, figsize=(11.0, 3.2 * len(pairs)), squeeze=False)
    for ax, (label, runs) in zip(axes[:, 0], pairs.items()):
        seed, site, integrated = next(r for r in runs if r[0] == PLOT_SEED)[:3]
        days = np.arange(1, len(site) + 1)
        ax.plot(days, integrated / 1e3, color=INTEGRATED_COLOR, linewidth=1.0, label="Integrated")
        ax.plot(days, site / 1e3, color=SITE_COLOR, linewidth=1.0, label="Site layer")
        ax.set_title(f"Daily Vector Flux — {label} (seed={seed})", fontsize=9, loc="left", pad=2)
        ax.set_ylabel("Vector flux [TJ/day]", fontsize=9)
        ax.set_xlim(days[0], days[-1])
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.legend(loc="lower right", fontsize=7, frameon=False)
    axes[-1, 0].set_xlabel("Day", fontsize=9)
    return _save(fig, "site_vs_integrated_daily_flux.png")


def plot_cumulative(pairs):
    fig, axes = plt.subplots(1, len(pairs), figsize=(5.5 * len(pairs), 3.8), squeeze=False)
    for ax, (label, runs) in zip(axes[0], pairs.items()):
        for i, (seed, site, integrated, _, _) in enumerate(runs):
            days = np.arange(1, len(site) + 1)
            ax.plot(days, np.cumsum(site - integrated) / 1e3, color=SEED_COLORS[i],
                    linewidth=1.4, label=f"seed={seed}")
        ax.axhline(0.0, color="0.35", linewidth=0.6)
        ax.set_title(f"Cumulative Flux Difference (site − integrated) — {label.split('-')[0]}",
                     fontsize=9, loc="left", pad=2)
        ax.set_xlabel("Day", fontsize=9)
        ax.set_ylabel("Cumulative difference [TJ]", fontsize=9)
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.legend(loc="best", fontsize=7, frameon=False)
    return _save(fig, "site_vs_integrated_cumulative.png")


def _save(fig, name):
    fig.tight_layout()
    path = STAGE4_DIR / name
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main():
    pairs = load_pairs()
    if not pairs:
        print(f"No site rollouts found in {STAGE4_DIR}")
        return

    lines = [f"Site layer vs integrated controller: vector_flux over applied hours; "
             f"'days >1% below' = site daily flux < {1 - DAILY_TOL:g} x integrated"]
    for label, runs in sorted(pairs.items()):
        lines += report(label, runs)

    plt.rcParams["font.family"] = "serif"
    paths = [plot_daily(pairs), plot_cumulative(pairs)]

    text_path = STAGE4_DIR / "site_compare.txt"
    text_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print()
    for path in [text_path] + paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
