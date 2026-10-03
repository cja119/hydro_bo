"""
Checks for the fundamental-floor mode of `PriceDynamics`.

Reports:
  1. theta for the basis parameters (config.yml price_dynamics block)
  2. with kappa_lambda = 0 (constant P_f): mean of price / P_f over
     N_PATHS paths x N_YEARS years daily, after a BURN_IN_YEARS burn-in,
     compensated (theta) and uncompensated (theta forced to 0)
  3. with the fundamental block absent: max |old - new| price over 365 days
     for the same seed, where "old" is `PriceDynamics` as committed at git HEAD
  4. `conditional_mean` vs Monte Carlo (FORECAST_PATHS paths of the existing
     step) from y_t = theta + d, d in FORECAST_OFFSETS: relative error and
     MC standard error at FORECAST_DAYS

Plots (to png/): lambda(t) and P_f(t); sample price paths with P_f(t);
histogram of price / P_f, compensated vs uncompensated; conditional-mean
forecast vs Monte Carlo mean and 5-95% band.
"""

import ast
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import yaml

from hydro_bo.envs.shipping.dynamics import PriceDynamics
from hydro_bo.utils.seeding import make_rng


ROOT_DIR = Path(__file__).resolve().parents[1]
DYNAMICS_REL = "src/hydro_bo/envs/shipping/dynamics.py"
OUT_DIR = ROOT_DIR / "png"

# Placeholder floor parameters for testing.
FUNDAMENTAL = {
    "h_g": 0.165,          # MMBtu/kg
    "P_NG": 10.0,          # $/MMBtu
    "k_g": 0.5,            # $/kg
    "e_g": 0.0095,         # tCO2/kg
    "lambda_0": 400.0,     # $/tCO2
    "lambda_inf": 250.0,   # $/tCO2
    "kappa_lambda": 0.2,   # 1/yr
}

DT = 1.0 / 365.0
N_PATHS = 2000
N_YEARS = 10
BURN_IN_YEARS = 1
N_SAMPLE_PATHS = 20
PLOT_YEARS = 20
CHECK_DAYS = 365
CHECK_SEED = 1234
FORECAST_PATHS = 20000
FORECAST_OFFSETS = [-0.3, 0.0, 0.3]
FORECAST_DAYS = [1, 7, 30, 90]
FORECAST_HORIZON_DAYS = 90

PATH_COLOR = "#2a78d6"
FLOOR_COLOR = "#eb6834"
UNCOMP_COLOR = "#eb6834"


def load_price_config():
    with open(ROOT_DIR / "src" / "hydro_bo" / "data" / "config.yml") as f:
        return yaml.safe_load(f).get("price_dynamics", {})


def make_price(pd_cfg, seed, fundamental=None, cls=PriceDynamics):
    kwargs = {} if fundamental is None else {"fundamental": fundamental}
    return cls(
        volatility=float(pd_cfg.get("sigma", 0.7)),
        long_run_log_price=float(pd_cfg.get("mu", 1.6094)),
        speed=float(pd_cfg.get("kappa", 5.0)),
        jump_intensity=float(pd_cfg.get("lambda", 12.0)),
        jump_mu=float(pd_cfg.get("jump_mu", 0.05)),
        jump_sigma=float(pd_cfg.get("jump_sigma", 0.2)),
        initial_price=float(pd_cfg.get("initial_price", 5.0)),
        seed=seed,
        **kwargs,
    )


def legacy_price_dynamics():
    """`PriceDynamics` class as committed at git HEAD."""
    src = subprocess.run(
        ["git", "show", f"HEAD:{DYNAMICS_REL}"],
        cwd=ROOT_DIR, capture_output=True, text=True, check=True,
    ).stdout
    node = next(n for n in ast.parse(src).body
                if isinstance(n, ast.ClassDef) and n.name == "PriceDynamics")
    namespace = {"np": np, "make_rng": make_rng}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<HEAD PriceDynamics>", "exec"), namespace)
    return namespace["PriceDynamics"]


def simulate_ratio(pd_cfg, fundamental, force_theta_zero=False):
    """price / P_f for each path and day after burn-in: (N_PATHS, n_days)."""
    n_steps = N_YEARS * 365
    burn = BURN_IN_YEARS * 365
    out = np.empty((N_PATHS, n_steps - burn))
    for i in range(N_PATHS):
        pd = make_price(pd_cfg, seed=i, fundamental=fundamental)
        if force_theta_zero:
            pd.theta = 0.0
            pd.y = 0.0
        for s in range(n_steps):
            price = pd.step(dt=DT)
            if s >= burn:
                out[i, s - burn] = price / pd.floor_price(pd.t)
    return out


def simulate_from(pd_cfg, offset):
    """Daily prices from y_t = theta + offset at t = 0, for FORECAST_PATHS
    paths of the existing step: (n_paths, FORECAST_HORIZON_DAYS + 1)."""
    out = np.empty((FORECAST_PATHS, FORECAST_HORIZON_DAYS + 1))
    for i in range(FORECAST_PATHS):
        pd = start_at(make_price(pd_cfg, seed=i, fundamental=FUNDAMENTAL), offset)
        out[i, 0] = pd.price
        for d in range(1, FORECAST_HORIZON_DAYS + 1):
            out[i, d] = pd.step(dt=DT)
    return out


def start_at(pd, offset):
    pd.y = pd.theta + offset
    pd.price = pd.floor_price(pd.t) * np.exp(pd.y)
    return pd


def forecast_curve(pd_cfg, offset):
    pd = start_at(make_price(pd_cfg, seed=0, fundamental=FUNDAMENTAL), offset)
    return np.array([pd.conditional_mean(d * DT) for d in range(FORECAST_HORIZON_DAYS + 1)])


def summarise_ratio(name, ratio):
    path_means = ratio.mean(axis=1)
    se = path_means.std(ddof=1) / np.sqrt(len(path_means))
    print(f"  {name:<28} mean = {ratio.mean():.4f}  (± {se:.4f} s.e. over paths)")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_floor(pd):
    t = np.linspace(0.0, PLOT_YEARS, PLOT_YEARS * 365 + 1)
    fig, (ax_l, ax_p) = plt.subplots(2, 1, figsize=(8.0, 5.0), sharex=True)
    ax_l.plot(t, pd.carbon_price(t), color=PATH_COLOR, linewidth=1.8)
    ax_l.set_title("Carbon Price λ(t)", fontsize=9, loc="left", pad=2)
    ax_l.set_ylabel("λ [$/tCO2]", fontsize=9)
    ax_p.plot(t, pd.floor_price(t), color=FLOOR_COLOR, linewidth=1.8)
    ax_p.set_title(r"Fundamental price $P_f(t)$", fontsize=9, loc="left", pad=2)
    ax_p.set_ylabel(r"$P_f$ [\$/kg]", fontsize=9)
    ax_p.set_xlabel("Time [years]", fontsize=9)
    for ax in (ax_l, ax_p):
        ax.grid(True, linewidth=0.4, alpha=0.5)
    ax_p.set_xlim(0, PLOT_YEARS)
    return _save(fig, "price_check_floor.png")


def plot_paths(pd_cfg):
    n_steps = PLOT_YEARS * 365
    t = np.arange(n_steps + 1) * DT
    fig, ax = plt.subplots(1, 1, figsize=(9.0, 4.2))
    for i in range(N_SAMPLE_PATHS):
        pd = make_price(pd_cfg, seed=i, fundamental=FUNDAMENTAL)
        prices = [pd.price] + [pd.step(dt=DT) for _ in range(n_steps)]
        ax.plot(t, prices, color=PATH_COLOR, linewidth=0.5, alpha=0.35,
                label="Price paths" if i == 0 else None)
    ax.plot(t, pd.floor_price(t), color=FLOOR_COLOR, linewidth=2.0, label=r"$P_f(t)$")
    ax.set_title(f"Hydrogen Price Paths ({N_SAMPLE_PATHS} samples, fundamental floor)",
                 fontsize=9, loc="left", pad=2)
    ax.set_xlabel("Time [years]", fontsize=9)
    ax.set_ylabel("Hydrogen price [$/kg]", fontsize=9)
    ax.set_xlim(0, PLOT_YEARS)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.legend(loc="upper right", fontsize=7, frameon=False)
    return _save(fig, "price_check_paths.png")


def plot_ratio_hist(comp, uncomp):
    hi = np.quantile(np.concatenate([comp.ravel(), uncomp.ravel()]), 0.999)
    bins = np.linspace(0.0, hi, 120)
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.0))
    for data, color, name in ((comp, PATH_COLOR, "compensated (θ)"),
                              (uncomp, UNCOMP_COLOR, "uncompensated (θ = 0)")):
        ax.hist(data.ravel(), bins=bins, density=True, histtype="step",
                color=color, linewidth=1.6, label=f"{name}, mean {data.mean():.3f}")
        ax.axvline(data.mean(), color=color, linestyle="--", linewidth=0.9)
    ax.axvline(1.0, color="0.35", linewidth=0.8)
    ax.set_title(r"Price / $P_f$ after burn-in ($\kappa_\lambda = 0$)", fontsize=9, loc="left", pad=2)
    ax.set_xlabel(r"Price / $P_f$ [-]", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.legend(loc="upper right", fontsize=7, frameon=False)
    return _save(fig, "price_check_ratio_hist.png")


def plot_forecast(curves):
    days = np.arange(FORECAST_HORIZON_DAYS + 1)
    fig, axes = plt.subplots(1, len(curves), figsize=(4.0 * len(curves), 3.8), sharey=True)
    for ax, (offset, forecast, mc) in zip(axes, curves):
        lo, hi = np.quantile(mc, [0.05, 0.95], axis=0)
        ax.fill_between(days, lo, hi, color=PATH_COLOR, alpha=0.15, linewidth=0,
                        label="MC 5-95%")
        ax.plot(days, mc.mean(axis=0), color=PATH_COLOR, linewidth=1.4, label="MC mean")
        ax.plot(days, forecast, color=FLOOR_COLOR, linestyle="--", linewidth=1.8,
                label="conditional_mean")
        ax.set_title(rf"$y_t = \theta {offset:+.1f}$", fontsize=9, loc="left", pad=2)
        ax.set_xlabel("Horizon h [days]", fontsize=9)
        ax.set_xlim(0, FORECAST_HORIZON_DAYS)
        ax.grid(True, linewidth=0.4, alpha=0.5)
    axes[0].set_ylabel("Hydrogen price [$/kg]", fontsize=9)
    axes[0].legend(loc="upper left", fontsize=7, frameon=False)
    return _save(fig, "price_check_forecast.png")


def _save(fig, name):
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    pd_cfg = load_price_config()
    plt.rcParams["font.family"] = "serif"

    # 1. theta for the basis parameters
    pd = make_price(pd_cfg, seed=0, fundamental=FUNDAMENTAL)
    diffusion_part = -pd.sigma**2 / (4 * pd.kappa)
    print("1. Compensator theta  "
          f"(sigma {pd.sigma:g}, kappa {pd.kappa:g}, lambda {pd.lambda_:g}, "
          f"jump_mu {pd.jump_mu:g}, jump_sigma {pd.jump_sigma:g})")
    print(f"  theta = {pd.theta:.6f}  (diffusion {diffusion_part:.6f}, "
          f"jumps {pd.theta - diffusion_part:.6f}); exp(-theta) = {np.exp(-pd.theta):.4f}")
    print(f"  P_f(0) = {pd.floor_price(0.0):.3f} $/kg, P_f(inf) = "
          f"{pd.floor_price(np.inf):.3f} $/kg, initial price = {pd.price:.3f} $/kg")

    # 2. constant floor: mean of price / P_f
    const_floor = dict(FUNDAMENTAL, kappa_lambda=0.0)
    print(f"\n2. kappa_lambda = 0: mean of price / P_f "
          f"({N_PATHS} paths x {N_YEARS} y daily, {BURN_IN_YEARS} y burn-in)")
    comp = simulate_ratio(pd_cfg, const_floor)
    summarise_ratio("compensated (theta)", comp)
    uncomp = simulate_ratio(pd_cfg, const_floor, force_theta_zero=True)
    summarise_ratio("uncompensated (theta = 0)", uncomp)

    # 3. block absent: new vs HEAD
    legacy = make_price(pd_cfg, seed=CHECK_SEED, cls=legacy_price_dynamics())
    new = make_price(pd_cfg, seed=CHECK_SEED)
    old_path = np.array([legacy.step(dt=DT) for _ in range(CHECK_DAYS)])
    new_path = np.array([new.step(dt=DT) for _ in range(CHECK_DAYS)])
    print(f"\n3. Block absent, seed {CHECK_SEED}, {CHECK_DAYS} days: "
          f"max |old - new| = {np.max(np.abs(old_path - new_path)):.3e} $/kg")

    # 4. conditional mean vs Monte Carlo
    print(f"\n4. conditional_mean vs Monte Carlo ({FORECAST_PATHS} paths, start t = 0)")
    print(f"  {'y_t':>10}  {'h [d]':>5}  {'forecast':>9}  {'MC mean':>9}  "
          f"{'rel err':>9}  {'MC s.e.':>8}")
    curves = []
    for offset in FORECAST_OFFSETS:
        forecast = forecast_curve(pd_cfg, offset)
        mc = simulate_from(pd_cfg, offset)
        curves.append((offset, forecast, mc))
        for d in FORECAST_DAYS:
            mean = mc[:, d].mean()
            se = mc[:, d].std(ddof=1) / np.sqrt(FORECAST_PATHS)
            print(f"  {f'theta{offset:+.1f}':>10}  {d:>5}  {forecast[d]:>9.4f}  {mean:>9.4f}  "
                  f"{forecast[d] / mean - 1:>+9.2%}  {se / mean:>8.2%}")

    print()
    for path in (plot_floor(pd), plot_paths(pd_cfg), plot_ratio_hist(comp, uncomp),
                 plot_forecast(curves)):
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
