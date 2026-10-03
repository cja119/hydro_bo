"""
Contract post-processing of the stage-0 monthly sent volumes.

Loads the `trajectory_log.py` .npz files in `paper_results/contract/stage0/`
and, per carrier, simulates a monthly delivery contract on the per-seed
monthly sent volumes V_m (each seed's 12 months in order, one independent
run per seed; results pooled over seeds x months):

  I_min = z * sigma_T * X / 30
  I_max = (n - 1) * Q_s            (zero-buffer case: I_max = I_min)
  I_0   = (I_min + I_max) / 2

  I_tilde = I_prev + V_m - X
  o = max(I_tilde - I_max, 0),  u = max(I_min - I_tilde, 0)
  I = I_tilde - o + u
  R = X*P + o*(1-eps)*P - u*(1+eps)*P

Each run's total revenue also gets P * (I_end - I_0): the change in buyer
tank inventory valued at the mid price (consignment stock, owned by the
exporter until the buyer consumes it). Summed over a run this gives
  sum R / P = sum V - eps * sum (o + u),
so the imbalance cost is exactly eps * mean(o + u) / mean(V), and for
eps = 0 mean revenue does not depend on X.

Prints the report (also written to `contract_postprocess.txt`) and saves
three figures, one panel per carrier, to the same directory.
"""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STAGE0_DIR = Path(__file__).resolve().parents[1] / "paper_results" / "contract" / "stage0"

PRICE = 5.0 * 1000.0            # $/t H2-eq (5 $/kg)
Z = 2.0
SIGMA_T_DAYS = 0.66
DAYS_PER_MONTH = 30.0
X_FRACS = np.arange(50, 121) / 100.0     # X / mean(V_m), steps of 1 %
EPSILONS = [0.0, 0.05, 0.1, 0.2]
TANK_NS = [1.5, 2.0, 3.0]
# Tank cases in display order: key, label.
TANK_CASES = [("zero", "zero buffer")] + [(n, f"n={n:g}") for n in TANK_NS]

PLOT_EPS = 0.1
CASE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
CASE_MARKERS = ["o", "s", "^", "D"]
REF_COLOR = "0.35"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_carriers():
    """carrier -> {"label", "seeds", "V" (n_seeds, n_months), "Q_s"}."""
    by_label = defaultdict(list)
    for path in sorted(STAGE0_DIR.glob("*-seed_*.npz")):
        d = np.load(path)
        if "monthly_sent_t_h2" not in d.files:
            print(f"Skipping {path.name}: written by an older trajectory_log.py")
            continue
        by_label[str(d["label"])].append(d)

    carriers = {}
    for label, runs in by_label.items():
        carrier = label.split("-")[0]
        if carrier in carriers:
            raise ValueError(f"More than one design for {carrier}: "
                             f"{carriers[carrier]['label']} and {label}")
        runs = sorted(runs, key=lambda d: int(d["seed"]))
        # Largest ship, same t H2-eq conversion as the profit equation.
        q_s = {float(np.max(d["ship_capacity"]) * d["calorific_value"] / 120.0) for d in runs}
        if len(q_s) != 1:
            raise ValueError(f"{label}: ship capacity differs across seeds: {q_s}")
        carriers[carrier] = {
            "label": label,
            "seeds": [int(d["seed"]) for d in runs],
            "V": np.stack([d["monthly_sent_t_h2"] for d in runs]),
            "Q_s": q_s.pop(),
        }
    return carriers


# ---------------------------------------------------------------------------
# Contract simulation
# ---------------------------------------------------------------------------

def tank_limits(case, X, q_s):
    i_min = Z * SIGMA_T_DAYS * X / DAYS_PER_MONTH
    if case == "zero":
        i_max = i_min.copy()
    else:
        i_max = np.full_like(X, (case - 1.0) * q_s)
    if np.any(i_max < i_min):
        raise ValueError(f"Tank case {case}: I_max < I_min for X >= {X[i_max < i_min].min():.0f}")
    return i_min, i_max


def simulate(V, X, i_min, i_max):
    """Run every seed independently over its months for every X.
    Returns o, u with shape (n_seeds, n_months, n_X) and I_0, I_end."""
    n_seeds, n_months = V.shape
    i_0 = (i_min + i_max) / 2.0
    inv = np.broadcast_to(i_0, (n_seeds, X.size)).copy()
    o = np.empty((n_seeds, n_months, X.size))
    u = np.empty_like(o)
    for m in range(n_months):
        i_tilde = inv + V[:, m, None] - X
        o[:, m] = np.maximum(i_tilde - i_max, 0.0)
        u[:, m] = np.maximum(i_min - i_tilde, 0.0)
        inv = i_tilde - o[:, m] + u[:, m]
    return {"o": o, "u": u, "I_0": i_0, "I_end": inv}


def revenue(X, o, u, eps):
    return X * PRICE + o * (1.0 - eps) * PRICE - u * (1.0 + eps) * PRICE


def best_x(mean_rev):
    """Index of the first maximiser, plus the index range of all X tied
    with it (the revenue curve is flat between order statistics of V)."""
    tied = np.flatnonzero(np.isclose(mean_rev, mean_rev.max(), rtol=1e-10, atol=0.0))
    return int(tied[0]), (int(tied[0]), int(tied[-1]))


def evaluate_carrier(c):
    V = c["V"]
    mean_v = V.mean()
    X = X_FRACS * mean_v
    n_months = V.shape[1]

    results = {}
    balance_err = 0.0
    spot_err = 0.0
    eps0_range = {}
    for case, _ in TANK_CASES:
        i_min, i_max = tank_limits(case, X, c["Q_s"])
        sim = simulate(V, X, i_min, i_max)
        o, u = sim["o"], sim["u"]
        d_inv = sim["I_end"] - sim["I_0"]                                    # (n_seeds, n_X)

        lhs = (o - u).sum(axis=1)
        rhs = (V[:, :, None] - X).sum(axis=1) - d_inv
        balance_err = max(balance_err, float(np.max(np.abs(lhs - rhs))))

        # Inventory change valued at P, spread over the run's months.
        inv_value = PRICE * d_inv.mean(axis=0) / n_months                   # (n_X,)
        mean_spot = (o + u).mean(axis=(0, 1))                                # (n_X,)

        for eps in EPSILONS:
            mean_rev = revenue(X, o, u, eps).mean(axis=(0, 1)) + inv_value  # (n_X,)
            spot_err = max(spot_err, float(np.max(np.abs(
                (mean_v * PRICE - mean_rev) - eps * PRICE * mean_spot))))
            if eps == 0.0:
                eps0_range[case] = float(mean_rev.max() - mean_rev.min())
            k, tie = best_x(mean_rev)
            results[case, eps] = {
                "mean_rev": mean_rev,
                "k": k,
                "tie": tie,
                "X_star": X[k],
                "cost": (mean_v * PRICE - mean_rev[k]) / (mean_v * PRICE),
                "gross_spot": mean_spot[k],
            }
    return {"X": X, "mean_v": mean_v, "median_v": float(np.median(V)),
            "balance_err": balance_err, "spot_err": spot_err,
            "eps0_range": eps0_range, "results": results}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _table(title, ev, fmt, flag_ties=False):
    lines = [title, "  " + f"{'':>12}" + "".join(f"{f'eps={e:g}':>11}" for e in EPSILONS)]
    for case, name in TANK_CASES:
        cells = []
        for eps in EPSILONS:
            r = ev["results"][case, eps]
            mark = "†" if flag_ties and r["tie"][1] > r["tie"][0] else " "
            cells.append(f"{fmt(r):>10}{mark}")
        lines.append("  " + f"{name:>12}" + "".join(cells))
    return lines


def report(carrier, c, ev):
    X, V = ev["X"], c["V"]
    lines = [
        f"\n=== {carrier}  ({c['label']}, seeds {c['seeds']}, {V.size} seed-months) ===",
        f"Q_s (largest ship) = {c['Q_s']:.1f} t H2-eq",
        f"mean V = {ev['mean_v']:.1f}, median V = {ev['median_v']:.1f} t H2-eq/month; "
        f"X grid {X[0]:.0f}..{X[-1]:.0f} ({len(X)} points)",
        "",
        "1. Volume balance  max |sum(o-u) - (sum(V-X) - (I_end-I_0))|"
        f" = {ev['balance_err']:.3e} t H2-eq",
        "   Spot cost check  max over n, eps, X of"
        f" |(mean(V)*P - mean R) - eps*P*mean(o+u)| = {ev['spot_err']:.3e} $/month",
        "   eps=0: max - min of mean revenue over X  "
        + ", ".join(f"{name} {ev['eps0_range'][case]:.3e}" for case, name in TANK_CASES)
        + " $/month",
        "",
        "2. Zero buffer: X* vs median(V), and mean R/P vs mean(V) - eps*mean|V-X*|",
        f"  {'eps':>5}  {'X*':>9}  {'X*/median':>9}  {'flat over X':>19}  "
        f"{'mean R/P':>9}  {'formula':>9}  {'diff':>9}",
    ]
    for eps in EPSILONS:
        r = ev["results"]["zero", eps]
        lo, hi = r["tie"]
        flat = f"{X[lo]:.0f}-{X[hi]:.0f}" if hi > lo else "-"
        rev_t = r["mean_rev"][r["k"]] / PRICE
        formula = ev["mean_v"] - eps * np.mean(np.abs(V - r["X_star"]))
        lines.append(f"  {eps:>5g}  {r['X_star']:>9.0f}  {r['X_star'] / ev['median_v']:>9.3f}  "
                     f"{flat:>19}  {rev_t:>9.1f}  {formula:>9.1f}  {rev_t - formula:>9.2e}")

    lines += [""] + _table("3. X* / median(V)", ev,
                           lambda r: f"{r['X_star'] / ev['median_v']:.3f}", flag_ties=True)
    lines += ["  † revenue tied over a range of X; X* is the lowest X in the range"]
    lines += [""] + _table("4. Imbalance cost (mean(V)*P - mean R) / (mean(V)*P) at X*", ev,
                           lambda r: f"{r['cost']:.2%}")
    return lines


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _save(fig, name):
    fig.tight_layout()
    path = STAGE0_DIR / name
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_revenue_vs_x(evals):
    fig, axes = plt.subplots(1, len(evals), figsize=(5.5 * len(evals), 3.8), squeeze=False)
    for ax, (carrier, ev) in zip(axes[0], evals.items()):
        for i, (case, name) in enumerate(TANK_CASES):
            r = ev["results"][case, PLOT_EPS]
            ax.plot(ev["X"] / 1e3, r["mean_rev"] / 1e6, color=CASE_COLORS[i], linewidth=1.6, label=name)
            ax.scatter(r["X_star"] / 1e3, r["mean_rev"][r["k"]] / 1e6, s=36, color=CASE_COLORS[i],
                       marker=CASE_MARKERS[i], edgecolors="white", linewidths=0.8, zorder=4)
        ax.axvline(ev["median_v"] / 1e3, color=REF_COLOR, linestyle="--", linewidth=0.8,
                   label="median(V)")
        ax.set_title(f"{carrier}: mean revenue vs commitment (eps={PLOT_EPS:g}, markers = X*)",
                     fontsize=9, loc="left", pad=2)
        ax.set_xlabel("Commitment X [kt H2-eq/month]", fontsize=9)
        ax.set_ylabel("Mean revenue [M$/month]", fontsize=9)
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.legend(loc="best", fontsize=7, frameon=False)
    return _save(fig, "contract-revenue_vs_X.png")


def plot_cost_vs_eps(evals):
    fig, axes = plt.subplots(1, len(evals), figsize=(5.5 * len(evals), 3.8), squeeze=False)
    for ax, (carrier, ev) in zip(axes[0], evals.items()):
        for i, (case, name) in enumerate(TANK_CASES):
            cost = [ev["results"][case, eps]["cost"] * 100 for eps in EPSILONS]
            ax.plot(EPSILONS, cost, color=CASE_COLORS[i], marker=CASE_MARKERS[i],
                    markersize=6, linewidth=1.6, label=name)
        ax.axhline(0.0, color=REF_COLOR, linewidth=0.6)
        ax.set_title(f"{carrier}: imbalance cost at X*", fontsize=9, loc="left", pad=2)
        ax.set_xlabel("Spot penalty eps [-]", fontsize=9)
        ax.set_ylabel("Imbalance cost [% of mean(V)·P]", fontsize=9)
        ax.set_xticks(EPSILONS)
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.legend(loc="best", fontsize=7, frameon=False)
    return _save(fig, "contract-imbalance_vs_eps.png")


def plot_spot_vs_n(evals):
    fig, axes = plt.subplots(1, len(evals), figsize=(5.5 * len(evals), 3.8), squeeze=False)
    pos = np.arange(len(TANK_CASES))
    for ax, (carrier, ev) in zip(axes[0], evals.items()):
        for i, eps in enumerate(EPSILONS):
            spot = [ev["results"][case, eps]["gross_spot"] / 1e3 for case, _ in TANK_CASES]
            ax.plot(pos, spot, color=CASE_COLORS[i], marker=CASE_MARKERS[i],
                    markersize=6, linewidth=1.6, label=f"eps={eps:g}")
        ax.set_xticks(pos)
        ax.set_xticklabels([name for _, name in TANK_CASES], fontsize=8)
        ax.set_title(f"{carrier}: mean gross spot volume (o+u) at X*", fontsize=9, loc="left", pad=2)
        ax.set_xlabel("Tank size case", fontsize=9)
        ax.set_ylabel("Spot volume [kt H2-eq/month]", fontsize=9)
        ax.set_ylim(bottom=0)
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.legend(loc="best", fontsize=7, frameon=False)
    return _save(fig, "contract-spot_volume_vs_n.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    carriers = load_carriers()
    if not carriers:
        print(f"No trajectory .npz files found in {STAGE0_DIR}")
        return

    lines = [f"P = {PRICE / 1000:g} $/kg, z = {Z:g}, sigma_T = {SIGMA_T_DAYS:g} d, "
             f"eps in {EPSILONS}, n in {TANK_NS}"]
    evals = {}
    for carrier, c in sorted(carriers.items()):
        evals[carrier] = evaluate_carrier(c)
        lines += report(carrier, c, evals[carrier])

    plt.rcParams["font.family"] = "serif"
    paths = [plot_revenue_vs_x(evals), plot_cost_vs_eps(evals), plot_spot_vs_n(evals)]

    text_path = STAGE0_DIR / "contract_postprocess.txt"
    text_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print()
    for p in [text_path] + paths:
        print(f"Saved: {p}")


if __name__ == "__main__":
    main()
