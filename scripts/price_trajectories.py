"""Plot sample hydrogen price trajectories from the price dynamics model."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from hydro_bo.envs.shipping.dynamics import PriceDynamics


N_TRAJECTORIES = 10
N_DAYS = 365
DT = 1.0 / 365.0


def load_price_config():
    cfg_path = Path(__file__).resolve().parents[1] / "src" / "hydro_bo" / "data" / "config.yml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("price_dynamics", {})


def simulate_trajectory(pd_cfg, seed):
    pd = PriceDynamics(
        volatility=float(pd_cfg.get("sigma", 0.7)),
        long_run_log_price=float(pd_cfg.get("mu", 1.6094)),
        speed=float(pd_cfg.get("kappa", 5.0)),
        jump_intensity=float(pd_cfg.get("lambda", 12.0)),
        jump_mu=float(pd_cfg.get("jump_mu", 0.05)),
        jump_sigma=float(pd_cfg.get("jump_sigma", 0.2)),
        initial_price=float(pd_cfg.get("initial_price", 5.0)),
        seed=seed,
    )
    prices = np.empty(N_DAYS + 1)
    prices[0] = pd.price
    for t in range(1, N_DAYS + 1):
        prices[t] = pd.step(dt=DT)
    return prices


def main():
    pd_cfg = load_price_config()
    initial_price = float(pd_cfg.get("initial_price", 5.0))
    long_run_price = float(np.exp(float(pd_cfg.get("mu", 1.6094))))

    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(1, 1, figsize=(9.0, 4.2))

    days = np.arange(N_DAYS + 1)
    colors = plt.cm.viridis(np.linspace(0.0, 0.9, N_TRAJECTORIES))

    for i in range(N_TRAJECTORIES):
        prices = simulate_trajectory(pd_cfg, seed=i)
        ax.plot(
            days,
            prices,
            color=colors[i],
            linewidth=1.4,
            alpha=0.9,
            label=f"seed={i}",
        )

    ax.axhline(
        long_run_price,
        color="k",
        linestyle="--",
        linewidth=0.8,
        alpha=0.6,
        label=f"long-run mean (${long_run_price:.2f}/kg)",
    )

    ax.set_title(
        f"Hydrogen Price Trajectories ({N_TRAJECTORIES} samples, dynamic price enabled)",
        fontsize=9,
        loc="left",
        pad=2,
    )
    ax.set_xlabel("Day", fontsize=9)
    ax.set_ylabel("Hydrogen price [$/kg]", fontsize=9)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.set_xlim(0, N_DAYS)
    ax.legend(loc="best", fontsize=7, ncols=2, frameon=False)

    fig.tight_layout()

    out_dir = Path(__file__).resolve().parents[1] / "png"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "price_trajectories.png"
    fig.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}  (initial_price=${initial_price}/kg)")


if __name__ == "__main__":
    main()
