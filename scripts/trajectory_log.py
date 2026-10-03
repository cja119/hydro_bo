"""
Log hourly ShippingEnv trajectories for a single ablation design.

Runs the unmodified `ShippingEnv('shipping')` for `--design` (default
`DEFAULT_DESIGN`) over `N_MONTHS` months per seed, set up the same way
`RayMultiMPC` workers set it up (see `hydro_bo.mpc.dispatcher.run_mpc`).
Each MPC solve commits a 24 h window: window 0 contributes t=0..24 and
every later window contributes t=1..24, because its t=0 is fixed to the
previous window's t=24. This gives one contiguous hourly record per
rollout.

Effective storage limits are capacity x backoff fraction, read off the
solved Pyomo instance. Note the MPC only enforces them for
t >= forecast_horizon, so the committed hours can sit outside them.

"Month" is the env's month (one `env.step` call; lengths follow
`Dynamics.get_forecasts`). Ships sent at a window's t=24 are attributed
to that window's month, matching the env's own sent-volume accounting.

Outputs (one per seed) in `paper_results/contract/stage0/`:
  - `<label>-seed_<n>.npz`  hourly arrays + per-month sent volume
  - `<label>-seed_<n>.png`  vector storage + departures / flux + curtailment
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import yaml
import numpy as np
from pyomo.environ import value

from hydro_bo import ShippingEnv
from hydro_bo.utils.logging_config import configure_logging, get_logger
from hydro_bo.utils.run_config import load_config, merge_env_overrides
from hydro_bo.utils.search_space import split_env_overrides

logger = get_logger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPTS_DIR.parent

# Highest BO objective among the NH3 ablation cells (1.88, 31/32 feasible).
DEFAULT_DESIGN = (
    ROOT_DIR / "paper_results" / "ablation" / "NH3"
    / "sobol_64_CI_NONE_comb_08" / "NH3-sobol_64_CI_NONE_comb_08-best.yml"
)
OUT_DIR = ROOT_DIR / "paper_results" / "contract" / "stage0"

SEEDS = [0, 1, 2]
N_MONTHS = 12
WINDOW_HOURS = 24

HOURLY_VARS = [
    "vector_flux",
    "energy_curtailed",
    "n_active_trains_conversion",
    "hydrogen_storage",
    "vector_storage",
]
STORAGE_VARS = ["hydrogen_storage", "vector_storage"]
SHIP_VARS = ["n_ship_sent", "n_ship_ordered"]
SCALAR_PARAMS = ["calorific_value", "conversion_trains_number"] + [
    f"{name}_{suffix}"
    for name in STORAGE_VARS
    for suffix in ("capacity", "lower_backoff", "upper_backoff")
]

SERIES_COLORS = ["#2a78d6", "#eb6834"]
SHIP_COLORS = {"small": "#eb6834", "medium": "#1baf7a", "large": "#eda100"}
SHIP_MARKERS = {"small": "o", "medium": "s", "large": "^"}
LIMIT_COLOR = "0.35"


def design_label(design_path: Path) -> str:
    return design_path.stem.removesuffix("-best")


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

class TrajectoryRecorder:
    """Wraps `MPCController.solve` so each committed window is read off
    the solved instance before the env hands off to the next window."""

    def __init__(self, controller, month_fn):
        self._controller = controller
        self._solve = controller.solve
        self._month_fn = month_fn
        self.ships = None
        self.params = None
        self.rows = []
        self.n_windows = 0

    def install(self):
        self._controller.solve = self

    def __call__(self, *args, **kwargs):
        out = self._solve(*args, **kwargs)
        self._record(self._controller.instance)
        self.n_windows += 1
        return out

    def _record_params(self, inst):
        self.ships = list(inst.ships)
        self.params = {name: float(value(getattr(inst, name))) for name in SCALAR_PARAMS}
        self.params["ship_capacity"] = np.array(
            [float(value(inst.ship_capacity[s])) for s in self.ships]
        )

    def _record(self, inst):
        if self.params is None:
            self._record_params(inst)

        grid1 = set(inst.grid1)
        h2_price = float(value(inst.h2_price))
        month = int(self._month_fn())
        t_first = 0 if self.n_windows == 0 else 1

        for t in range(t_first, WINDOW_HOURS + 1):
            row = {
                "hour": self.n_windows * WINDOW_HOURS + t,
                "window": self.n_windows,
                "month": month,
                "h2_price": h2_price,
            }
            for name in HOURLY_VARS:
                row[name] = float(value(getattr(inst, name)[t]))
            for name in SHIP_VARS:
                var = getattr(inst, name)
                row[name] = [
                    int(round(value(var[s, t]))) if t in grid1 else 0
                    for s in self.ships
                ]
            self.rows.append(row)

    def to_arrays(self):
        keys = self.rows[0].keys()
        data = {k: np.array([r[k] for r in self.rows]) for k in keys}
        data["ships"] = np.array(self.ships)
        data.update(self.params)

        # Effective limits: capacity x backoff, broadcast per hour.
        n_hours = len(data["hour"])
        for name in STORAGE_VARS:
            cap = self.params[f"{name}_capacity"]
            for side in ("lower", "upper"):
                data[f"{name}_{side}"] = np.full(n_hours, cap * self.params[f"{name}_{side}_backoff"])

        # Sent volume in t H2-eq, same conversion as the profit equation:
        # n_ship_sent * ship_capacity * calorific_value / 120.
        sent_t_h2 = data["n_ship_sent"] @ self.params["ship_capacity"] * self.params["calorific_value"] / 120.0
        months = np.arange(1, N_MONTHS + 1)
        data["sent_t_h2"] = sent_t_h2
        data["months"] = months
        data["monthly_sent_t_h2"] = np.array([sent_t_h2[data["month"] == m].sum() for m in months])
        data["monthly_hours"] = np.array([(data["month"] == m).sum() for m in months])
        return data


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def build_env_args(design):
    """Same env_args + param split that `multi_mpc.py` feeds the dispatcher."""
    cfg = load_config(SCRIPTS_DIR / "config.yml", vector_override=design["vector"])
    params, env_overrides = split_env_overrides(design)
    env_args = merge_env_overrides(cfg, env_overrides)
    return env_args, params


def _deep_set(target, updates):
    for k, v in updates.items():
        if isinstance(v, dict):
            if k not in target or not isinstance(target[k], dict):
                target[k] = {}
            _deep_set(target[k], v)
        else:
            target[k] = v


def run_rollout(env_args, params, seed):
    env = ShippingEnv("shipping")
    with env as e:
        _deep_set(e, env_args.get("config", {}))
        e["seed"] = seed
        e["mpc"]["planning_model"] = dict(params)

    recorder = TrajectoryRecorder(env._controller, month_fn=lambda: env._dynamics._call_count)
    recorder.install()

    while True:
        observation, reward, done, info = env.step(None)
        call_count = info.get("call_count")
        logger.info("trajectory_log.progress", seed=seed, month=call_count,
                    total=N_MONTHS, windows=recorder.n_windows)
        if done or (call_count is not None and call_count >= N_MONTHS):
            break

    return recorder.to_arrays()


def save_npz(data, design_path, seed, path):
    np.savez(
        path,
        seed=seed,
        label=design_label(design_path),
        design_path=str(design_path.relative_to(ROOT_DIR)),
        **data,
    )
    logger.info("trajectory_log.npz_saved", path=str(path), n_hours=len(data["hour"]),
                monthly_sent_t_h2=data["monthly_sent_t_h2"].round(1).tolist())


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_trajectory(npz_path, png_path):
    import matplotlib.pyplot as plt

    d = np.load(npz_path)
    hours = d["hour"]
    ships = [str(s) for s in d["ships"]]

    plt.rcParams["font.family"] = "serif"
    fig, (ax_vec, ax_en) = plt.subplots(2, 1, figsize=(12.0, 6.5), sharex=True)

    # --- Top: vector storage, capacity, effective limits, departures ---
    ax_vec.plot(hours, d["vector_storage"], color=SERIES_COLORS[0],
                linewidth=1.2, zorder=3, label="Vector storage")
    ax_vec.axhline(float(d["vector_storage_capacity"]), color=LIMIT_COLOR,
                   linewidth=0.9, zorder=2, label="Capacity")
    ax_vec.plot(hours, d["vector_storage_upper"], color=LIMIT_COLOR, linestyle="--",
                linewidth=0.8, alpha=0.8, zorder=1, label="Effective limits (capacity × backoff)")
    ax_vec.plot(hours, d["vector_storage_lower"], color=LIMIT_COLOR, linestyle="--",
                linewidth=0.8, alpha=0.8, zorder=1)

    sent = d["n_ship_sent"]
    for j, size in enumerate(ships):
        idx = np.flatnonzero(sent[:, j] > 0)
        if idx.size == 0:
            continue
        ax_vec.scatter(
            hours[idx], d["vector_storage"][idx],
            s=36, marker=SHIP_MARKERS.get(size, "o"),
            color=SHIP_COLORS.get(size, "k"), edgecolors="white", linewidths=0.8,
            zorder=4, label=f"Departure ({size})  Σ={int(sent[:, j].sum())}",
        )

    ax_vec.set_title("Stored Vector", fontsize=9, loc="left", pad=2)
    ax_vec.set_ylabel("Mass Stored [kt]", fontsize=9)
    ax_vec.grid(True, linewidth=0.4, alpha=0.5)
    ax_vec.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7, frameon=False)

    # --- Bottom: vector flux and curtailment (both GJ/h) ---
    ax_en.plot(hours, d["vector_flux"], color=SERIES_COLORS[0],
               linewidth=0.9, zorder=3, label="Vector flux")
    ax_en.plot(hours, d["energy_curtailed"], color=SERIES_COLORS[1],
               linewidth=0.9, alpha=0.85, zorder=2, label="Energy curtailed")
    ax_en.set_title("Conversion Throughput and Curtailment", fontsize=9, loc="left", pad=2)
    ax_en.set_ylabel("Energy [GJ/h]", fontsize=9)
    ax_en.set_xlabel("Time [h]", fontsize=9)
    ax_en.grid(True, linewidth=0.4, alpha=0.5)
    ax_en.set_xlim(hours[0], hours[-1])
    ax_en.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7, frameon=False)

    fig.suptitle(f"{d['label']}  (seed={int(d['seed'])})", fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    logger.info("trajectory_log.plot_saved", path=str(png_path))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    design_path = Path(args.design).resolve()
    label = design_label(design_path)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seeds_tag = "_".join(str(s) for s in args.seeds)
    configure_logging(log_file=OUT_DIR / f"{label}-seeds_{seeds_tag}.log")

    with open(design_path) as f:
        design = yaml.safe_load(f)
    env_args, params = build_env_args(design)
    logger.info("trajectory_log.start", design=str(design_path), seeds=args.seeds,
                months=N_MONTHS, env_config=env_args["config"])

    for seed in args.seeds:
        stem = f"{label}-seed_{seed}"
        npz_path = OUT_DIR / f"{stem}.npz"
        if not args.plot_only:
            data = run_rollout(env_args, params, seed)
            save_npz(data, design_path, seed, npz_path)
        plot_trajectory(npz_path, OUT_DIR / f"{stem}.png")

    logger.info("trajectory_log.complete", out_dir=str(OUT_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log hourly ShippingEnv trajectories for one design.")
    parser.add_argument("--design", type=str, default=str(DEFAULT_DESIGN),
                        help="Path to a *-best.yml design (default: the NH3 comb_08 design).")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                        help=f"Env master seeds, one rollout each (default: {SEEDS}).")
    parser.add_argument("--plot_only", action="store_true",
                        help="Skip the rollouts and re-plot from existing .npz files.")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
