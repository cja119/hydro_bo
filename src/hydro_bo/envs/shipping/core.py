"""
This module contains the shipping environment which interacts with the model predictive
control optimiser for the shipping problem.
"""

from __future__ import annotations

from hydro_bo.algs.logging_config import get_logger
from hydro_bo.algs.mpc import MPCController, MPCSolveError
from .utils import (
    PlotTheme,
    apply_theme,
    args_dict,
    import_mpc_data,
    import_mpc_functions,
    plot_energy_management_figure,
    plot_scheduling_figure,
    status_message,
)
from .dynamics import Dynamics


logger = get_logger(__name__)


class ShippingEnv:

    def __init__(self) -> None:
        self.idx = 0
        self._args = args_dict()
        self._controller = MPCController()
        self._slow_data = {
            "params": {
                "storage_capacity": 10,
                "mean_ship_transit_time": 35,
                "std_ship_transit_time": 2,
            }
        }
        self._prev_rollover_snapshot = None

    def __enter__(self) -> None:
        return self._args

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._build_env()

    def _build_env(self) -> None:
        # rebuild controller/dynamics from current args
        self._controller = MPCController()
        self._controller_data = import_mpc_data(
            self._args["mpc"]["planning_model"],
            self._args["vector"],
            self._args["mpc"]["random_param"],
        )

        self._controller_data.update(
            import_mpc_functions(
                self._args["mpc"]["data_folder"], self._controller_data["sets"]
            )
        )

        # Allow env args to override price_dynamics settings from config.yml
        if "price_dynamics" in self._args:
            self._controller_data["price_dynamics"].update(self._args["price_dynamics"])

        self._controller.build(self._controller_data)
        self._dynamics = Dynamics(self._controller_data, self._slow_data, self._args)
        self._state = self._dynamics.get_state()
        self._prev_rollover_snapshot = None

    def render(self, mode="human"):
        return self._controller.render()

    def reset(self) -> None:
        self._build_env()

    def step(self, action, verbose=False):
        n_steps = None
        iter_idx = None
        last_mpc_args = None
        stored_val = 0

        try:
            observation = {}
            n_steps = self._dynamics.get_forecasts()
            total_sent = 0
            t_sent_vol = 0
            prev_profit = 0
            if self._dynamics.results is not None:
                prev_profit = self._dynamics.results.get(("actual_profit", 0), 0)

            for i in range(n_steps):
                iter_idx = i

                # Update the state with the new demand
                mpc_args = self._dynamics.get_mpc_args()
                last_mpc_args = mpc_args
                self._controller.update(mpc_args, self._dynamics.results)

                # Solve the mpc controller
                results, self._dynamics.states, delta_stored, sent_vol = (
                    self._controller.solve(solver=self._args["mpc"]["solver"])
                )

                self._dynamics.set_results(results)

                total_sent += sum(
                    self._dynamics.states["sent_ship"][s]
                    for s in self._controller_data["sets"]["ships"]
                )
                t_sent_vol += sent_vol
                stored_val += delta_stored

                # Print the current iteration and the total ships sent
                if verbose:
                    status_message(i, self._dynamics.iter_count // n_steps)

                # Simulate the shipping dynamics with the new demand
                self._dynamics.simulate_shipping_dynamics()

                # Keep the previous solve's next-day forecast for failure comparison.
                self._prev_rollover_snapshot = self._build_rollover_snapshot(mpc_args)

                self._state = self._dynamics.get_state()

                # Update the observation with the current state
                observation["destination_storage"] = self._dynamics.destination_storage
                observation["ship_destination"] = self._dynamics.ship_destination

            # k$ + t * 5 k$/t / (t) = k$ / t or $/kg <- This is the profit per/kg
            profit_inc = results.get(("actual_profit", 0), 0) - prev_profit
            reward = (profit_inc * 1000) + stored_val * self._dynamics.h2_price
            observation["total_tonnes"] = stored_val + t_sent_vol

            call_count = getattr(self._dynamics, "_call_count", None)

            if self._dynamics.iter_count // n_steps >= self._args["months"]:
                if verbose:
                    print(
                        "\r[INFO] Maximum iterations reached. Total reward: ", end=" " * 20
                    )
                return observation, reward, True, {"call_count": call_count}
            else:
                if verbose:
                    print(f"\r[REWARD] M$ {reward:.4f}", end="" * 50)
                return observation, reward, False, {"call_count": call_count}

        except MPCSolveError as exc:
            self._log_solve_failure_context(
                error=exc,
                attempt=0,
                n_steps=n_steps,
                iter_idx=iter_idx,
                mpc_args=last_mpc_args,
            )
            raise

    def _log_solve_failure_context(self, error, attempt, n_steps, iter_idx, mpc_args):
        expected_nonzero = []
        arrived_now = {}
        carryover_t0 = None
        if mpc_args is not None:
            arrived_now = dict(mpc_args["ship_arrived"]["param"]["initialize"])
            expected_init = mpc_args["expected_ships"]["param"]["initialize"]
            expected_nonzero = [
                {"ship": ship, "hour": int(hour), "count": int(count)}
                for (ship, hour), count in expected_init.items()
                if count != 0
            ]
            expected_nonzero.sort(key=lambda row: (row["hour"], row["ship"]))

        # Snapshot key carryover states from previous successful MPC solve.
        # This helps diagnose infeasibilities caused by inconsistent t=0 fixing.
        if self._dynamics.results is not None:
            try:
                ships = self._controller_data["sets"]["ships"]
                carryover_t0 = {
                    "cumulative_charge_t0": float(self._dynamics.results.get(("cumulative_charge", 0), 0.0)),
                    "ship_charge_rate_t0": float(self._dynamics.results.get(("ship_charge_rate", 0), 0.0)),
                    "waiting_ships_t0": {
                        s: float(self._dynamics.results.get(("waiting_ships", (s, 0)), 0.0))
                        for s in ships
                    },
                }
            except Exception:
                carryover_t0 = None

        tracking = self._dynamics.ship_tracking_snapshot()
        rollover_compare = self._compare_with_previous_rollover(mpc_args)

        logger.error(
            "shipping_mpc_solve_failed",
            attempt=attempt,
            month=getattr(self._dynamics, "_call_count", None),
            iter_count=self._dynamics.iter_count,
            inner_iter=iter_idx,
            n_steps=n_steps,
            termination_condition=str(getattr(error, "termination_condition", "unknown")),
            message=str(getattr(error, "message", "")),
            ship_arrived_now=arrived_now,
            expected_ship_nonzero_count=len(expected_nonzero),
            expected_ship_nonzero_preview=expected_nonzero[:20],
            dynamics_ship_tracking=tracking,
            ordered_ship_state=dict(self._dynamics.states.get("ordered_ship", {})),
            sent_ship_state=dict(self._dynamics.states.get("sent_ship", {})),
            carryover_t0=carryover_t0,
            rollover_compare=rollover_compare,
        )

    def _build_rollover_snapshot(self, mpc_args):
        if mpc_args is None:
            return None

        wind = list(mpc_args["energy_wind"]["param"]["initialize"])
        expected = mpc_args["expected_ships"]["param"]["initialize"]

        ships = self._controller_data["sets"]["ships"]

        return {
            "energy_next_day": wind[24:48],
            "expected_next_day_by_ship": {
                s: int(expected.get((s, 24), 0)) for s in ships
            },
            "expected_day2_by_ship": {
                s: int(expected.get((s, 48), 0)) for s in ships
            },
        }

    def _compare_with_previous_rollover(self, mpc_args):
        if self._prev_rollover_snapshot is None or mpc_args is None:
            return None

        wind = list(mpc_args["energy_wind"]["param"]["initialize"])
        current_arrived = dict(mpc_args["ship_arrived"]["param"]["initialize"])
        expected = mpc_args["expected_ships"]["param"]["initialize"]
        ships = self._controller_data["sets"]["ships"]

        prev_next_wind = self._prev_rollover_snapshot["energy_next_day"]
        curr_today_wind = wind[:24]

        n = min(len(prev_next_wind), len(curr_today_wind))
        wind_abs_diff_sum = float(
            sum(abs(prev_next_wind[i] - curr_today_wind[i]) for i in range(n))
        )
        wind_abs_diff_max = float(
            max((abs(prev_next_wind[i] - curr_today_wind[i]) for i in range(n)), default=0.0)
        )

        prev_expected_next_day = self._prev_rollover_snapshot["expected_next_day_by_ship"]
        prev_expected_day2 = self._prev_rollover_snapshot["expected_day2_by_ship"]
        curr_expected_day1 = {s: int(expected.get((s, 24), 0)) for s in ships}

        return {
            "weather_prev_nextday_vs_curr_today": {
                "sum_abs_diff": wind_abs_diff_sum,
                "max_abs_diff": wind_abs_diff_max,
            },
            "ships_prev_expected_nextday_vs_curr_arrived_today": {
                s: {
                    "prev_expected_nextday": int(prev_expected_next_day.get(s, 0)),
                    "curr_arrived_today": int(current_arrived.get(s, 0)),
                    "delta": int(current_arrived.get(s, 0)) - int(prev_expected_next_day.get(s, 0)),
                }
                for s in ships
            },
            "ships_prev_expected_day2_vs_curr_expected_day1": {
                s: {
                    "prev_expected_day2": int(prev_expected_day2.get(s, 0)),
                    "curr_expected_day1": int(curr_expected_day1.get(s, 0)),
                    "delta": int(curr_expected_day1.get(s, 0)) - int(prev_expected_day2.get(s, 0)),
                }
                for s in ships
            },
        }


class ShippingEnvPlot(ShippingEnv):
    """
    Plotting variant of ShippingEnv.

    - Mirrors ShippingEnv step logic (mpc update/solve, dynamics simulate, reward/done calc).
    - Adds plotting overheads via self._controller.visualise_output(...)
    - step(...) is a generator that yields figures each internal iteration so notebooks can
      animate via `for fig in env.step(None): display(fig)`.
    - The final (observation, reward, done, info) for the *last* call to step is stored in
      `self.last_transition` for optional inspection.
    """

    def __init__(self) -> None:
        self.idx = 0
        self._args = args_dict()
        self._controller = MPCController()
        self._slow_data = {
            "params": {
                "storage_capacity": 10,
                "mean_ship_transit_time": 35,
                "std_ship_transit_time": 2,
            }
        }
        self._prev_rollover_snapshot = None
        self.last_transition = None  # (observation, reward, done, info)

    def __enter__(self) -> None:
        return self._args

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._controller_data = import_mpc_data(
            self._args["mpc"]["planning_model"],
            self._args["vector"],
            self._args["mpc"]["random_param"],
        )

        self._controller_data.update(
            import_mpc_functions(
                self._args["mpc"]["data_folder"], self._controller_data["sets"]
            )
        )

        # Allow env args to override price_dynamics settings from config.yml
        if "price_dynamics" in self._args:
            self._controller_data["price_dynamics"].update(self._args["price_dynamics"])

        self._controller.build(self._controller_data)
        self._dynamics = Dynamics(self._controller_data, self._slow_data, self._args)
        self._state = self._dynamics.get_state()

    def render(self, mode="human"):
        return self._controller.render()

    def reset(self) -> None:
        self._build_env()

    def plot_data(self):
        return self._controller._joined_data

    def plot_figures(
        self,
        *,
        theme: PlotTheme | None = None,
        schedule_figsize=(12, 14),
        energy_figsize=(12, 12),
        max_ticks: int = 10,
    ):
        """Render scheduling and energy management figures from collected plot data."""

        if not hasattr(self._controller, "_joined_data") or self._controller._joined_data is None:
            raise ValueError("No plot data available; run step() with plot_horizon first to populate joined_data.")

        theme_to_use = theme or PlotTheme()
        apply_theme(theme_to_use)

        joined_data = self._controller._joined_data
        fig_sched, _ = plot_scheduling_figure(
            joined_data,
            figsize=schedule_figsize,
            max_ticks=max_ticks,
        )
        fig_energy, _ = plot_energy_management_figure(
            joined_data,
            figsize=energy_figsize,
            max_ticks=max_ticks,
        )

        return fig_sched, fig_energy

    def step(self, action, verbose: bool = False, plot_horizon: int = 24):
        """
        Generator step:
          - yields a matplotlib figure each internal MPC step (so you can animate).
          - stores final transition in `self.last_transition`.

        Parameters
        ----------
        action : unused (kept for API compatibility)
        verbose : bool
        plot_horizon : int
            Horizon passed to self._controller.visualise_output(plot_horizon)
        """
        n_steps = None
        iter_idx = None
        last_mpc_args = None

        try:
            observation = {}

            n_steps = self._dynamics.get_forecasts()

            total_sent = 0
            t_sent_vol = 0
            prev_profit = 0

            if self._dynamics.results is not None:
                prev_profit = self._dynamics.results.get(("actual_profit", 0), 0)

            results = None
            stored_val = 0
            sent_vol = 0

            for i in range(n_steps):
                iter_idx = i
                # Update the state with the new demand
                mpc_args = self._dynamics.get_mpc_args()
                last_mpc_args = mpc_args
                self._controller.update(mpc_args, self._dynamics.results)

                # Solve the mpc controller (match ShippingEnvV2 signature/behavior)
                results, self._dynamics.states, stored_val, sent_vol = (
                    self._controller.solve(
                        solver=self._args["mpc"]["solver"],
                    )
                )

                self._dynamics.set_results(results)

                total_sent += sum(
                    self._dynamics.states["sent_ship"][s]
                    for s in self._controller_data["sets"]["ships"]
                )
                t_sent_vol += sent_vol

                if verbose:
                    status_message(i, self._dynamics.iter_count // n_steps)

                # Plotting overhead
                self._controller.visualise_output(plot_horizon)

                # Simulate the shipping dynamics with the new demand
                self._dynamics.simulate_shipping_dynamics()
                self._prev_rollover_snapshot = self._build_rollover_snapshot(mpc_args)
                self._state = self._dynamics.get_state()

                observation["destination_storage"] = self._dynamics.destination_storage
                observation["ship_destination"] = self._dynamics.ship_destination

                yield self._controller.render()

            # Log per-type relaxation usage for this simulation run
            self._controller.log_relaxation_summary()

            profit_inc = 0
            if results is not None:
                profit_inc = results.get(("actual_profit", 0), 0) - prev_profit

            reward = (profit_inc * 1000) + stored_val * self._dynamics.h2_price
            observation["total_tonnes"] = stored_val + t_sent_vol

            done = self._dynamics.iter_count // n_steps >= self._args["months"]
            info = {"call_count": getattr(self._dynamics, "_call_count", None)}

            self.last_transition = (observation, reward, done, info)

        except MPCSolveError as exc:
            self._log_solve_failure_context(
                error=exc,
                attempt=0,
                n_steps=n_steps,
                iter_idx=iter_idx,
                mpc_args=last_mpc_args,
            )
            raise
