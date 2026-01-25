"""
This module contains the shipping environment which performs a bilevel model predictive
control optimisation for the shipping problem.
"""

from __future__ import annotations

from algs.mpc.core import MPCController
from .utils import import_mpc_data, import_mpc_functions, args_dict, status_message
from .dynamics import Dynamics


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

    def __enter__(self) -> None:
        return self._args

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._controller_data = import_mpc_data(
            self._args["mpc"]["data_folder"],
            self._args["mpc"]["planning_model"],
            self._args["vector"],
            self._args["mpc"]["random_param"],
        )

        self._controller_data.update(
            import_mpc_functions(
                self._args["mpc"]["data_folder"], self._controller_data["sets"]
            )
        )

        self._controller.build(self._controller_data)
        self._dynamics = Dynamics(self._controller_data, self._slow_data, self._args)
        self._state = self._dynamics.get_state()

    def render(self, mode="human"):
        return self._controller.render()

    def reset(self) -> None:
        with self as slf:
            pass

    def step(self, action, verbose=False):
        observation = {}
        n_steps = self._dynamics.get_forecasts()
        total_sent = 0
        t_sent_vol = 0
        prev_profit = 0
        if self._dynamics.results is not None:
            prev_profit = self._dynamics.results.get(("actual_profit", 0), 0)

        for i in range(n_steps):

            # Update the state with the new demand
            mpc_args = self._dynamics.get_mpc_args()
            self._controller.update(mpc_args, self._dynamics.results)

            # Solve the mpc controller
            results, self._dynamics.states, stored_val, sent_vol = (
                self._controller.solve(True, solver=self._args["mpc"]["solver"])
            )

            self._dynamics.set_results(results)

            total_sent += sum(
                self._dynamics.states["sent_ship"][s]
                for s in self._controller_data["sets"]["ships"]
            )
            t_sent_vol += sent_vol

            # Print the curent iteration and the total ships sent
            if verbose:
                status_message(i, self._dynamics.iter_count // n_steps)

            # Simulate the shipping dynamics with the new demand
            self._dynamics.simulate_shipping_dynamics(new_demand)

            self._state = self._dynamics.get_state()

            # Update the observation with the current state
            observation["destination_storage"] = self._dynamics.destination_storage
            observation["ship_destination"] = self._dynamics.ship_destination

        # k$ + t * 5 k$/t / (t) = k$ / t or $/kg <- This is the profit per/kg
        profit_inc = results.get(("actual_profit", 0), 0) - prev_profit
        reward = (profit_inc * 1000) + stored_val * 5
        observation["total_tonnes"] = stored_val + t_sent_vol

        if self._dynamics.iter_count // n_steps >= self._args["months"]:
            if verbose:
                print(
                    "\r[INFO] Maximum iterations reached. Total reward: ", end=" " * 20
                )
            return observation, reward, True, {}
        else:
            if verbose:
                print(f"\r[REWARD] M$ {reward:.4f}", end="" * 50)
            return observation, reward, False, {}


class ShippingEnvPlot:
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

        self._controller.build(self._controller_data)
        self._dynamics = Dynamics(self._controller_data, self._slow_data, self._args)
        self._state = self._dynamics.get_state()

    def render(self, mode="human"):
        return self._controller.render()

    def reset(self) -> None:
        with self as slf:
            pass

    def plot_data(self):
        return self._controller._joined_data

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
            # Update the state with the new demand
            mpc_args = self._dynamics.get_mpc_args()
            self._controller.update(mpc_args, self._dynamics.results)

            # Solve the mpc controller (match ShippingEnvV2 signature/behavior)
            results, self._dynamics.states, stored_val, sent_vol = (
                self._controller.solve(
                    True,
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
            # (Assumes .visualise_output internally updates matplotlib state.)
            self._controller.visualise_output(plot_horizon)

            # Simulate the shipping dynamics with the new demand
            self._dynamics.simulate_shipping_dynamics()
            self._state = self._dynamics.get_state()

            # Update the observation with the current state
            observation["destination_storage"] = self._dynamics.destination_storage
            observation["ship_destination"] = self._dynamics.ship_destination

            # Yield a figure each internal step for notebook animation
            yield self._controller.render()

        # Final reward/termination logic (match ShippingEnvV2)
        profit_inc = 0
        if results is not None:
            profit_inc = results.get(("actual_profit", 0), 0) - prev_profit

        reward = (profit_inc * 1000) + stored_val * 5
        observation["total_tonnes"] = stored_val + t_sent_vol

        done = self._dynamics.iter_count // n_steps >= self._args["months"]
        info = {}

        self.last_transition = (observation, reward, done, info)
