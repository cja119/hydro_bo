"""
Utility functions for algorithms
"""

import contextlib
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from pyomo.environ import value


# ---- Bayesopt utils ---- #

def ensure_dirs(tmp_dir: Optional[Path] = None, checkpoint_dir: Optional[Path] = None):
    base_tmp = (
        Path(tmp_dir)
        if tmp_dir is not None
        else Path(__file__).parent.parent.parent / "tmp"
    )
    bayesopt_dir = base_tmp / "bayesopt"
    ray_results_dir = base_tmp / "ray_results"
    ckpt_dir = (
        Path(checkpoint_dir) if checkpoint_dir is not None else base_tmp / "checkpoints"
    )
    for path in (bayesopt_dir, ray_results_dir, ckpt_dir):
        path.mkdir(parents=True, exist_ok=True)
    return base_tmp, bayesopt_dir, ray_results_dir, ckpt_dir


def parse_memory_string(memory_str: str) -> int:
    memory_str = memory_str.upper()
    if memory_str.endswith("GB"):
        return int(float(memory_str[:-2]) * 1024 * 1024 * 1024)
    if memory_str.endswith("MB"):
        return int(float(memory_str[:-2]) * 1024 * 1024)
    if memory_str.endswith("KB"):
        return int(float(memory_str[:-2]) * 1024)
    return int(memory_str)


# ---- MPC utils ---- #

@contextlib.contextmanager
def suppress_output(supress: bool = True):
    if not supress:
        yield
        return

    with open(os.devnull, "w") as devnull:
        old_out, old_err = sys.stdout, sys.stderr
        gurobi_logger = logging.getLogger("gurobipy")
        old_gurobi_disabled = gurobi_logger.disabled
        old_gurobi_level = gurobi_logger.level
        sys.stdout, sys.stderr = devnull, devnull
        gurobi_logger.disabled = True
        gurobi_logger.setLevel(logging.CRITICAL + 1)
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err
            gurobi_logger.disabled = old_gurobi_disabled
            gurobi_logger.setLevel(old_gurobi_level)


def ext_visualise_output(
    solve,
    axs,
    run_count,
    time_step=24,
    joined_data=None,
    fig=None,
):
    import matplotlib.pyplot as plt

    """
    Output visualisation for external use with provided axes.
    """

    plt.rcParams["font.family"] = "serif"

    axs = np.asarray(axs, dtype=object).ravel()
    if len(axs) < 4:
        raise ValueError(f"ext_visualise_output needs 4 axes, got {len(axs)}")
    axs = axs[:4]

    # Define colors
    NAVY = "#001f3f"
    BURGUNDY = "#800020"
    TITLE_PAD = 10

    # Helper to style each plot
    def plot_style(ax, x, y, title, ylabel):
        ax.scatter(x, y, color=NAVY, s=18, alpha=0.25, zorder=2, label="Value")
        ax.plot(x, y, color=BURGUNDY, linewidth=2, alpha=0.9, zorder=3, label="Value")
        ax.set_title(title, pad=TITLE_PAD)
        ax.set_ylabel(ylabel)
        ax.grid(True)

    # Helpers for time series contiguity
    steps = list(range(run_count * time_step, (run_count + 1) * time_step + 1))
    shifted = range(0, time_step + 1)
    daily_shifted = range(0, time_step + 1, 24)
    daily_steps = list(
        range(run_count * time_step, (run_count + 1) * time_step + 1, 24)
    )

    # Extract data from solve
    ships = getattr(solve, "ships", [])
    n_ordered = {
        s: [value(getattr(solve, "n_ship_ordered")[s, t]) for t in daily_shifted]
        for s in ships
    }
    n_ship_sent = {
        s: [value(getattr(solve, "n_ship_sent")[s, t]) for t in daily_shifted]
        for s in ships
    }

    vector_storage = [value(getattr(solve, "vector_storage")[t]) for t in shifted]
    cumulative_charge = [value(getattr(solve, "cumulative_charge")[t]) for t in shifted]

    # Helper functions to extract series and scalars
    def series(name, default=float("nan")):
        var = getattr(solve, name, None)
        if var is None:
            return [default for _ in shifted]
        return [value(var[t]) for t in shifted]

    def scalar(name, default=float("nan")):
        var = getattr(solve, name, None)
        if var is None:
            return default
        return float(value(var))

    # Extract series
    energy_turbine = series("energy_wind")
    vector_flux = series("vector_flux")
    n_active_trains_conversion = series("n_active_trains_conversion")
    energy_curtailed = series("energy_curtailed")
    energy_compression = series("energy_compression")
    energy_electrolysis = series("energy_electrolysis")
    energy_conversion = series("energy_conversion")
    energy_fuelcell = series("energy_fuelcell")
    energy_wind_profile = series("energy_wind")
    energy_solar_profile = series("energy_solar")

    # Compute renewable supply
    ren_cap = scalar("renewable_energy_capacity")
    if np.isfinite(ren_cap):
        renewable_supply_wind = [x * ren_cap for x in energy_wind_profile]
        renewable_supply_solar = [x * ren_cap for x in energy_solar_profile]
    else:
        renewable_supply_wind = [float("nan") for _ in shifted]
        renewable_supply_solar = [float("nan") for _ in shifted]

    # Extract hydrogen series
    hydrogen_storage = series("hydrogen_storage")
    hydrogen_produced = series("hydrogen_produced")
    hydrogen_used = series("hydrogen_used")
    hydrogen_stored = series("hydrogen_stored")
    hydrogen_removed = series("hydrogen_removed")
    hydrogen_consumed_fuelcell = series("hydrogen_consumed_fuelcell")
    hydrogen_storage_delta = [float("nan")] + [
        hydrogen_storage[i] - hydrogen_storage[i - 1]
        for i in range(1, len(hydrogen_storage))
    ]

    # Combine with previous runs unless first run
    if joined_data is None or run_count == 0:
        joined_data = {
            "steps": steps[:],
            "daily_steps": daily_steps[:],
            "n_ordered": {s: n_ordered[s][:] for s in ships},
            "n_ship_sent": {s: n_ship_sent[s][:] for s in ships},
            "vector_storage": vector_storage[:],
            "cumulative_charge": cumulative_charge[:],
            "energy_turbine": energy_turbine[:],
            "n_active_trains_conversion": n_active_trains_conversion[:],
            "vector_flux": vector_flux[:],
            "energy_curtailed": energy_curtailed[:],
            "energy_compression": energy_compression[:],
            "energy_electrolysis": energy_electrolysis[:],
            "energy_conversion": energy_conversion[:],
            "energy_fuelcell": energy_fuelcell[:],
            "energy_wind_profile": energy_wind_profile[:],
            "energy_solar_profile": energy_solar_profile[:],
            "renewable_supply_wind": renewable_supply_wind[:],
            "renewable_supply_solar": renewable_supply_solar[:],
            "hydrogen_storage": hydrogen_storage[:],
            "hydrogen_produced": hydrogen_produced[:],
            "hydrogen_used": hydrogen_used[:],
            "hydrogen_stored": hydrogen_stored[:],
            "hydrogen_removed": hydrogen_removed[:],
            "hydrogen_consumed_fuelcell": hydrogen_consumed_fuelcell[:],
            "hydrogen_storage_delta": hydrogen_storage_delta[:],
        }
    else:
        joined_data["steps"].extend(steps)
        joined_data["daily_steps"].extend(daily_steps)
        joined_data["vector_storage"].extend(vector_storage)
        joined_data["cumulative_charge"].extend(cumulative_charge)
        joined_data.setdefault("energy_turbine", []).extend(energy_turbine)
        joined_data.setdefault("n_active_trains_conversion", []).extend(
            n_active_trains_conversion
        )
        joined_data.setdefault("vector_flux", []).extend(vector_flux)
        joined_data.setdefault("energy_curtailed", []).extend(energy_curtailed)
        joined_data.setdefault("energy_compression", []).extend(energy_compression)
        joined_data.setdefault("energy_electrolysis", []).extend(energy_electrolysis)
        joined_data.setdefault("energy_conversion", []).extend(energy_conversion)
        joined_data.setdefault("energy_fuelcell", []).extend(energy_fuelcell)
        joined_data.setdefault("energy_wind_profile", []).extend(energy_wind_profile)
        joined_data.setdefault("energy_solar_profile", []).extend(energy_solar_profile)
        joined_data.setdefault("renewable_supply_wind", []).extend(
            renewable_supply_wind
        )
        joined_data.setdefault("renewable_supply_solar", []).extend(
            renewable_supply_solar
        )
        joined_data.setdefault("hydrogen_storage", []).extend(hydrogen_storage)
        joined_data.setdefault("hydrogen_produced", []).extend(hydrogen_produced)
        joined_data.setdefault("hydrogen_used", []).extend(hydrogen_used)
        joined_data.setdefault("hydrogen_stored", []).extend(hydrogen_stored)
        joined_data.setdefault("hydrogen_removed", []).extend(hydrogen_removed)
        joined_data.setdefault("hydrogen_consumed_fuelcell", []).extend(
            hydrogen_consumed_fuelcell
        )
        joined_data.setdefault("hydrogen_storage_delta", []).extend(
            hydrogen_storage_delta
        )

        for s in ships:
            joined_data["n_ordered"][s].extend(n_ordered[s])
            joined_data["n_ship_sent"][s].extend(n_ship_sent[s])

    # Clear axes
    for ax in axs:
        ax.cla()

    # Plot ships ordered / sent by capacity
    ax_ships, ax_vec, ax_fill, ax_energy = axs

    ship_list = list(joined_data["n_ordered"].keys())
    caps = {}
    for s in ship_list:
        try:
            caps[s] = float(value(getattr(solve, "ship_capacity")[s]))
        except Exception:
            caps[s] = np.nan

    # Bin ships by capacity tertiles
    cap_vals = np.array([v for v in caps.values() if np.isfinite(v)], dtype=float)
    if cap_vals.size == 0:
        bins = {"Small": [], "Medium": ship_list, "Large": []}
    else:
        q1, q2 = np.quantile(cap_vals, [1 / 3, 2 / 3])
        bins = {"Small": [], "Medium": [], "Large": []}
        for s in ship_list:
            c = caps.get(s, np.nan)
            if not np.isfinite(c):
                bins["Medium"].append(s)
            elif c <= q1:
                bins["Small"].append(s)
            elif c <= q2:
                bins["Medium"].append(s)
            else:
                bins["Large"].append(s)

    # Plot ships ordered / sent by capacity
    size_marker = {"Small": "o", "Medium": "^", "Large": "s"}
    ship_marker_size = {"Small": 55, "Medium": 75, "Large": 95}
    ordered_alpha = {"Small": 0.45, "Medium": 0.55, "Large": 0.65}
    sent_alpha = {"Small": 0.45, "Medium": 0.55, "Large": 0.65}
    ordered_color = {"Small": "#b85a72", "Medium": BURGUNDY, "Large": "#4d0010"}
    sent_color = {"Small": "#4c6a86", "Medium": NAVY, "Large": "#001022"}

    # Plot each bin
    for size_name, ships_in_bin in bins.items():
        if not ships_in_bin:
            continue

        length = len(joined_data["daily_steps"])
        y_ord = np.zeros(length, dtype=float)
        y_sent = np.zeros(length, dtype=float)

        for s in ships_in_bin:
            y_ord += np.asarray(joined_data["n_ordered"][s], dtype=float)
            y_sent += np.asarray(joined_data["n_ship_sent"][s], dtype=float)

        ax_ships.scatter(
            joined_data["daily_steps"],
            y_ord,
            color=ordered_color[size_name],
            s=ship_marker_size[size_name],
            alpha=0.25,
            marker=size_marker[size_name],
            edgecolors="none",
            zorder=2,
        )
        ax_ships.scatter(
            joined_data["daily_steps"],
            y_sent,
            color=sent_color[size_name],
            s=ship_marker_size[size_name],
            alpha=0.25,
            marker=size_marker[size_name],
            edgecolors="none",
            zorder=2,
        )

        ax_ships.plot(
            joined_data["daily_steps"],
            y_ord,
            color=ordered_color[size_name],
            linestyle="-",
            linewidth=2.0,
            alpha=ordered_alpha[size_name],
            zorder=3,
            label=f"Ordered ({size_name})",
        )
        ax_ships.plot(
            joined_data["daily_steps"],
            y_sent,
            color=sent_color[size_name],
            linestyle="--",
            linewidth=2.0,
            alpha=sent_alpha[size_name],
            zorder=3,
            label=f"Sent ({size_name})",
        )
    # Final plot styling
    ax_ships.set_title("Ships Ordered / Sent (by Capacity)", pad=TITLE_PAD)
    ax_ships.set_ylabel("Count")
    ax_ships.grid(True)

    handles, labels = ax_ships.get_legend_handles_labels()
    uniq = {}
    for handle, label in zip(handles, labels):
        if label not in uniq:
            uniq[label] = handle
    ax_ships.legend(
        list(uniq.values()),
        list(uniq.keys()),
        loc="upper left",
        fontsize=8,
        ncols=2,
        frameon=False,
    )

    # Plot vector storage
    plot_style(
        ax_vec,
        joined_data["steps"],
        joined_data["vector_storage"],
        "Stored Vector",
        "[kt]",
    )
    ax_vec.legend(loc="upper left", fontsize=8, frameon=False)

    # Plot ship fill level
    plot_style(
        ax_fill,
        joined_data["steps"],
        [x / 1000 for x in joined_data["cumulative_charge"]],
        "Ship Fill",
        "Mass (H2-eq) [kt]",
    )
    ax_fill.legend(loc="upper left", fontsize=8, frameon=False)

    # Plot energy turbine output
    plot_style(
        ax_energy,
        joined_data["steps"],
        joined_data["energy_turbine"],
        "Single Turbine Energy",
        "Energy [GJ/h]",
    )
    ax_energy.legend(loc="upper left", fontsize=8, frameon=False)

    # Final axis adjustments
    ax_energy.set_xlabel("Time [h]")
    for ax in (ax_ships, ax_vec, ax_fill):
        ax.tick_params(labelbottom=False)

    xs = joined_data["steps"]
    if len(xs) > 10:
        ax_energy.set_xticks(list(np.linspace(xs[0], xs[-1], 10, dtype=int)))

    # Adjust layout
    if fig is not None:
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.subplots_adjust(hspace=0.45)
        fig.canvas.draw()

    return joined_data
