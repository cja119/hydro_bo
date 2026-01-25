"""
Utils file for the inner loop of the model predictive control (MPC) algorithm.
"""

import yaml
from pathlib import Path
from pyomo.environ import value
import sys, os
import contextlib
import numpy as np

# matplotlib import moved to function level to avoid import issues


def add_equations(model, environment_name: str) -> None:
    """
    Adds the equations to the model.
    """

    current_path = Path(__file__).parent
    data_path = (
        current_path.parent.parent / "data/shipping/" + environment_name + "/fast_loop"
    )


@contextlib.contextmanager
def suppress_output(supress: bool = True):
    if supress:
        with open(os.devnull, "w") as devnull:
            old_out, old_err = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = devnull, devnull
            try:
                yield
            finally:
                sys.stdout, sys.stderr = old_out, old_err
    else:
        yield
        pass

def ext_visualise_output(
    solve,
    axs,
    run_count,
    time_step=24,
    joined_data=None,
    fig=None,
):
    import numpy as np
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "serif"

    NAVY = "#001f3f"
    BURGUNDY = "#800020"

    axs = np.asarray(axs, dtype=object).ravel()
    if len(axs) < 4:
        raise ValueError(f"ext_visualise_output needs 4 axes, got {len(axs)}")
    axs = axs[:4]

    TITLE_PAD = 10

    def plot_style(ax, x, y, title, ylabel, ylim=None, scatter=True):
        if scatter:
            ax.scatter(x, y, color=NAVY, s=18, alpha=0.25, zorder=2, label="Value")
        ax.plot(x, y, color=BURGUNDY, linewidth=2, alpha=0.9, zorder=3, label="Value")
        ax.set_title(title, pad=TITLE_PAD)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        if ylim is not None:
            ax.set_ylim(*ylim)

    try:
        value  # noqa: F821
    except NameError:
        from pyomo.environ import value  # type: ignore

    steps = list(range(run_count * time_step, (run_count + 1) * time_step + 1))
    shifted = range(0, time_step + 1)

    ships = getattr(solve, "ships", None)
    daily_shifted = range(0, time_step + 1, 24)
    daily_steps = list(range(run_count * time_step, (run_count + 1) * time_step + 1, 24))

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

    # -----------------------------
    # Helpers for diagnostic series
    # -----------------------------
    def series(name, default=float("nan")):
        """Get a time series [t in shifted] from solve.<name>[t], else NaNs."""
        try:
            _var = getattr(solve, name)
            return [value(_var[t]) for t in shifted]
        except Exception:
            return [default for _ in shifted]

    def scalar(name, default=float("nan")):
        """Get a scalar value from solve.<name>, else NaN."""
        try:
            return float(value(getattr(solve, name)))
        except Exception:
            return default

    # -----------------------------
    # Core plotted series (existing)
    # -----------------------------
    # Default plot remains "Single Turbine Energy" using wind profile if present
    energy_turbine = series("energy_wind")

    # vector_flux (graceful fallback if missing)
    vector_flux = series("vector_flux")

    # n_active_trains_conversion (graceful fallback if missing)
    n_active_trains_conversion = series("n_active_trains_conversion")

    # -----------------------------------------
    # NEW: Energy-related series (all relevant)
    # -----------------------------------------
    energy_curtailed = series("energy_curtailed")
    energy_compression = series("energy_compression")
    energy_electrolysis = series("energy_electrolysis")
    energy_conversion = series("energy_conversion")
    energy_fuelcell = series("energy_fuelcell")

    # Renewable profiles (Params) - keep both even if only one used
    energy_wind_profile = series("energy_wind")
    energy_solar_profile = series("energy_solar")

    # Optional: compute renewable supply in same units as balance term
    ren_cap = scalar("renewable_energy_capacity", default=float("nan"))
    if np.isfinite(ren_cap):
        renewable_supply_wind = [x * ren_cap for x in energy_wind_profile]
        renewable_supply_solar = [x * ren_cap for x in energy_solar_profile]
    else:
        renewable_supply_wind = [float("nan") for _ in shifted]
        renewable_supply_solar = [float("nan") for _ in shifted]

    # --------------------------------------------
    # NEW: Hydrogen-flow-related series (all flows)
    # --------------------------------------------
    hydrogen_storage = series("hydrogen_storage")
    hydrogen_produced = series("hydrogen_produced")
    hydrogen_used = series("hydrogen_used")
    hydrogen_stored = series("hydrogen_stored")
    hydrogen_removed = series("hydrogen_removed")
    hydrogen_consumed_fuelcell = series("hydrogen_consumed_fuelcell")

    # Handy derived diagnostics (optional, but useful)
    # net hydrogen change per hour (model should satisfy balance with removed/stored etc.)
    # NOTE: first element uses nan because we don't have t-1 in shifted start
    hydrogen_storage_delta = [float("nan")] + [
        hydrogen_storage[i] - hydrogen_storage[i - 1] for i in range(1, len(hydrogen_storage))
    ]

    # -------------------------------------------------------
    # Build / extend joined_data (append across rolling runs)
    # -------------------------------------------------------
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
            # --- Energy diagnostics ---
            "energy_curtailed": energy_curtailed[:],
            "energy_compression": energy_compression[:],
            "energy_electrolysis": energy_electrolysis[:],
            "energy_conversion": energy_conversion[:],
            "energy_fuelcell": energy_fuelcell[:],
            "energy_wind_profile": energy_wind_profile[:],
            "energy_solar_profile": energy_solar_profile[:],
            "renewable_supply_wind": renewable_supply_wind[:],
            "renewable_supply_solar": renewable_supply_solar[:],
            # --- Hydrogen diagnostics ---
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
        joined_data.setdefault("n_active_trains_conversion", []).extend(n_active_trains_conversion)
        joined_data.setdefault("vector_flux", []).extend(vector_flux)

        # --- Energy diagnostics ---
        joined_data.setdefault("energy_curtailed", []).extend(energy_curtailed)
        joined_data.setdefault("energy_compression", []).extend(energy_compression)
        joined_data.setdefault("energy_electrolysis", []).extend(energy_electrolysis)
        joined_data.setdefault("energy_conversion", []).extend(energy_conversion)
        joined_data.setdefault("energy_fuelcell", []).extend(energy_fuelcell)
        joined_data.setdefault("energy_wind_profile", []).extend(energy_wind_profile)
        joined_data.setdefault("energy_solar_profile", []).extend(energy_solar_profile)
        joined_data.setdefault("renewable_supply_wind", []).extend(renewable_supply_wind)
        joined_data.setdefault("renewable_supply_solar", []).extend(renewable_supply_solar)

        # --- Hydrogen diagnostics ---
        joined_data.setdefault("hydrogen_storage", []).extend(hydrogen_storage)
        joined_data.setdefault("hydrogen_produced", []).extend(hydrogen_produced)
        joined_data.setdefault("hydrogen_used", []).extend(hydrogen_used)
        joined_data.setdefault("hydrogen_stored", []).extend(hydrogen_stored)
        joined_data.setdefault("hydrogen_removed", []).extend(hydrogen_removed)
        joined_data.setdefault("hydrogen_consumed_fuelcell", []).extend(hydrogen_consumed_fuelcell)
        joined_data.setdefault("hydrogen_storage_delta", []).extend(hydrogen_storage_delta)

        for s in ships:
            joined_data["n_ordered"][s].extend(n_ordered[s])
            joined_data["n_ship_sent"][s].extend(n_ship_sent[s])

    # -----------------------------
    # Plotting (unchanged)
    # -----------------------------
    for ax in axs:
        ax.cla()

    ax_ships, ax_vec, ax_fill, ax_energy = axs

    ship_list = list(joined_data["n_ordered"].keys())

    cap = {}
    for s in ship_list:
        try:
            cap[s] = float(value(getattr(solve, "ship_capacity")[s]))
        except Exception:
            cap[s] = np.nan

    caps = np.array([cap[s] for s in ship_list if np.isfinite(cap[s])], dtype=float)
    if caps.size == 0:
        bins = {"Small": [], "Medium": ship_list, "Large": []}
    else:
        q1, q2 = np.quantile(caps, [1 / 3, 2 / 3])
        bins = {"Small": [], "Medium": [], "Large": []}
        for s in ship_list:
            c = cap.get(s, np.nan)
            if not np.isfinite(c):
                bins["Medium"].append(s)
            elif c <= q1:
                bins["Small"].append(s)
            elif c <= q2:
                bins["Medium"].append(s)
            else:
                bins["Large"].append(s)

    size_marker = {"Small": "o", "Medium": "^", "Large": "s"}
    ship_marker_size = {"Small": 55, "Medium": 75, "Large": 95}
    ordered_alpha = {"Small": 0.45, "Medium": 0.55, "Large": 0.65}
    sent_alpha = {"Small": 0.45, "Medium": 0.55, "Large": 0.65}
    scatter_alpha = 0.25
    lw = 2.0
    ordered_color = {"Small": "#b85a72", "Medium": BURGUNDY, "Large": "#4d0010"}
    sent_color = {"Small": "#4c6a86", "Medium": NAVY, "Large": "#001022"}

    for size_name, ships_in_bin in bins.items():
        if len(ships_in_bin) == 0:
            continue

        L = len(joined_data["daily_steps"])
        y_ord = np.zeros(L, dtype=float)
        y_sent = np.zeros(L, dtype=float)

        for s in ships_in_bin:
            y_ord += np.asarray(joined_data["n_ordered"][s], dtype=float)
            y_sent += np.asarray(joined_data["n_ship_sent"][s], dtype=float)

        ax_ships.scatter(
            joined_data["daily_steps"],
            y_ord,
            color=ordered_color[size_name],
            s=ship_marker_size[size_name],
            alpha=scatter_alpha,
            marker=size_marker[size_name],
            edgecolors="none",
            zorder=2,
        )
        ax_ships.scatter(
            joined_data["daily_steps"],
            y_sent,
            color=sent_color[size_name],
            s=ship_marker_size[size_name],
            alpha=scatter_alpha,
            marker=size_marker[size_name],
            edgecolors="none",
            zorder=2,
        )

        ax_ships.plot(
            joined_data["daily_steps"],
            y_ord,
            color=ordered_color[size_name],
            linestyle="-",
            linewidth=lw,
            alpha=ordered_alpha[size_name],
            zorder=3,
            label=f"Ordered ({size_name})",
        )
        ax_ships.plot(
            joined_data["daily_steps"],
            y_sent,
            color=sent_color[size_name],
            linestyle="--",
            linewidth=lw,
            alpha=sent_alpha[size_name],
            zorder=3,
            label=f"Sent ({size_name})",
        )

    ax_ships.set_title("Ships Ordered / Sent (by Capacity)", pad=TITLE_PAD)
    ax_ships.set_ylabel("Count")
    ax_ships.grid(True)

    handles, labels = ax_ships.get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        if l not in uniq:
            uniq[l] = h
    ax_ships.legend(
        list(uniq.values()),
        list(uniq.keys()),
        loc="upper left",
        fontsize=8,
        ncols=2,
        frameon=False,
    )

    plot_style(
        ax_vec,
        joined_data["steps"],
        joined_data["vector_storage"],
        title="Stored Vector",
        ylabel="[kt]",
        scatter=True,
    )
    ax_vec.legend(loc="upper left", fontsize=8, frameon=False)

    plot_style(
        ax_fill,
        joined_data["steps"],
        [x / 1000 for x in joined_data["cumulative_charge"]],
        title="Ship Fill",
        ylabel="Mass (H2-eq) [kt]",
        scatter=True,
    )
    ax_fill.legend(loc="upper left", fontsize=8, frameon=False)

    plot_style(
        ax_energy,
        joined_data["steps"],
        joined_data["energy_turbine"],
        title="Single Turbine Energy",
        ylabel="Energy [GJ/h]",
        scatter=True,
    )
    ax_energy.legend(loc="upper left", fontsize=8, frameon=False)

    ax_energy.set_xlabel("Time [h]")
    for ax in (ax_ships, ax_vec, ax_fill):
        ax.tick_params(labelbottom=False)

    max_ticks = 10
    xs = joined_data["steps"]
    if len(xs) > max_ticks:
        xticks = list(np.linspace(xs[0], xs[-1], max_ticks, dtype=int))
        ax_energy.set_xticks(xticks)

    if fig is not None:
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.subplots_adjust(hspace=0.45)
        fig.canvas.draw()

    return joined_data

