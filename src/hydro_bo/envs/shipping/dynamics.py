from numpy.random import normal
from pyomo.environ import value
from meteor_py import GetData
from math import floor
from .utils import temporal_align, generate_weather_forecast, count_and_shift_arrivals

import numpy as np

class Dynamics:
    def __init__(self, fast_data, slow_data, args):
        self._fast_data = fast_data
        self._slow_data = slow_data
        self._call_count = 0
        self._args = args

        np.random.seed(args["seed"])

        self._init_weather_data(args["seed"])

        self.new_states = None
        self.iter_count = 0
        self.idx = 0
        self.results = None

        self.states = self._init_state_variables()
        self.destination_storage = 0.5 * slow_data["params"]["storage_capacity"]

        # Price dynamics (OU + jump diffusion) — toggled via config.yml price_dynamics.enabled
        pd_cfg = fast_data.get("price_dynamics", {})
        if pd_cfg.get("enabled", False):
            self._price_dynamics = PriceDynamics(
                volatility=float(pd_cfg.get("sigma", 0.7)),
                long_run_log_price=float(pd_cfg.get("mu", 1.6094)),
                speed=float(pd_cfg.get("kappa", 5.0)),
                jump_intensity=float(pd_cfg.get("lambda", 12.0)),
                jump_mu=float(pd_cfg.get("jump_mu", 0.05)),
                jump_sigma=float(pd_cfg.get("jump_sigma", 0.2)),
                initial_price=float(pd_cfg.get("initial_price", 5.0)),
                seed=args["seed"],
            )
            self.h2_price = float(pd_cfg.get("initial_price", 5.0))
        else:
            self._price_dynamics = None
            self.h2_price = 5.0

        self.ship_destination = {s: [] for s in fast_data["sets"]["ships"]}
        self.ship_origin = {s: [] for s in fast_data["sets"]["ships"]}
        self.ship_origin_latent = {s: [] for s in fast_data["sets"]["ships"]}
        self.ship_origin_baseline = {s: [] for s in fast_data["sets"]["ships"]}
        self.expected_arrivals = {s: [] for s in fast_data["sets"]["ships"]}
        self.expected_destinations = {s: [] for s in fast_data["sets"]["ships"]}

    def _init_weather_data(self, seed):
        weather_file = self._args["weather_data"]["weather_file"]
        self._weather_data = GetData([weather_file]).data()
        self._weather_data = temporal_align(self._weather_data, randomise=True, seed=seed)

    def _init_state_variables(self):
        p = self._fast_data["params"]
        sets = self._fast_data["sets"]
        return {
            "current_ships": 1,
            "hydrogen_storage": 0.5 * p["hydrogen_storage_capacity"],
            "vector_storage": 0.5 * p["vector_storage_capacity"],
            "energy_conversion": (
                0.5
                * p["conversion_trains_number"]
                * p["variable_energy_penalty_conversion"]
                * p["single_train_limit_conversion"]
            ),
            "cumulative_charge": 0,
            "ordered_ship": {size: 0 for size in sets["ships"]},
            "sent_ship": {size: 0 for size in sets["ships"]},
        }

    def get_state(self):
        return (
            self.iter_count,
            self.states,
            self.destination_storage,
            self.ship_destination,
            self.ship_origin,
            self.expected_arrivals,
            self.expected_destinations,
            self.results,
        )

    def get_forecasts(self):
        DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        self._call_count += 1
        return DAYS_IN_MONTH[self._call_count % 12]

    def get_mpc_args(self):
        weather_forecast = generate_weather_forecast(
            self._weather_data, self.idx, self._fast_data["sets"]["grid0"]
        )

        origin_arrive = {}
        expected_arrivals_indexed = {
            (s, t): 0
            for s in self._fast_data["sets"]["ships"]
            for t in self._fast_data["sets"]["grid1"]
        }
        grid_hours = set(self._fast_data["sets"]["grid1"])

        for size in self.ship_origin:
            origin_arrive[size] = count_and_shift_arrivals(
                self.ship_origin,
                self.expected_arrivals,
                size,
                self.ship_origin_latent,
                self.ship_origin_baseline,
            )
            if origin_arrive[size] < 0:
                raise ValueError(
                    f"Invalid negative ship_arrived count for {size}: {origin_arrive[size]}"
                )

            # expected_ships carries absolute expected arrivals from in-transit
            # ships. Keep this nonnegative so port-capacity balances cannot be
            # forced by synthetic negative arrivals.
            noisy_counts = {}
            for arrival in self.expected_arrivals[size]:
                hour = int(arrival) * 24
                if hour in grid_hours:
                    noisy_counts[hour] = noisy_counts.get(hour, 0) + 1

            for hour in grid_hours:
                count = noisy_counts.get(hour, 0)
                if count != 0:
                    expected_arrivals_indexed[size, hour] += count

        return {
            "energy_wind": {
                "name": "energy_wind",
                "loc": "exogenous",
                "param": {
                    "set": self._fast_data["sets"]["grid0"],
                    "initialize": weather_forecast,
                },
            },
            "ship_arrived": {
                "name": "ship_arrived",
                "loc": "exogenous",
                "param": {
                    "set": self._fast_data["sets"]["ships"],
                    "initialize": origin_arrive,
                },
            },
            "expected_ships": {
                "name": "expected_ships",
                "loc": "exogenous",
                "param": {
                    "set": (self._fast_data["sets"]["ships"], self._fast_data["sets"]["grid1"]),
                    "initialize": expected_arrivals_indexed,
                },
            },
            "h2_price": {
                "name": "h2_price",
                "loc": "exogenous",
                "param": {
                    "set": None,
                    "initialize": self.h2_price,
                },
            },
        }

    def simulate_shipping_dynamics(self):
        self._simulate_origin_shipping()
        self._decrement_ship_counters()
        if self._price_dynamics is not None:
            self.h2_price = float(self._price_dynamics.step(dt=1 / 365))
        self.idx += 24
        self.iter_count += 1

    def _simulate_origin_shipping(self):
        p = self._fast_data["params"]
        for size, num_orders in self.states["ordered_ship"].items():
            if num_orders <= 0:
                continue

            # New ships start with deterministic mean arrival time.
            base_arrival_days = max(1, int(round(value(p["mean_ship_arrival_time"]))))
            arrivals = [base_arrival_days for _ in range(int(num_orders))]
            self.ship_origin[size].extend(arrivals)
            self.ship_origin_latent[size].extend(float(v) for v in arrivals)
            self.ship_origin_baseline[size].extend(arrivals)
            self.expected_arrivals[size].extend(
                [base_arrival_days] * int(num_orders)
            )

    def _decrement_ship_counters(self):
        p = self._fast_data["params"]
        std_days = float(value(p["std_ship_arrival_time"]))

        for size in self._fast_data["sets"]["ships"]:
            updated_latent = []
            updated_origin = []
            updated_baseline = []
            for remaining_days in self.ship_origin_latent[size]:
                # Freeze noise for near-arrivals to avoid one-day rollover jitter.
                disturbance_days = 0.0 if float(remaining_days) <= 1.0 else float(normal(0.0, std_days))
                new_remaining_latent = float(remaining_days) - 1.0 + disturbance_days
                updated_latent.append(new_remaining_latent)

                # Arrival accounting rule:
                # ETAs below 1 day are treated as realised arrivals (0);
                # ETAs above 1 day remain expected and are rounded up.
                if new_remaining_latent < 1.0:
                    discrete_remaining = 0
                else:
                    discrete_remaining = max(0, int(floor(new_remaining_latent)))
                updated_origin.append(discrete_remaining)

            for remaining_days in self.ship_origin_baseline[size]:
                new_remaining = float(remaining_days) - 1.0
                if new_remaining < 1.0:
                    discrete_remaining = 0
                else:
                    discrete_remaining = max(0, int(floor(new_remaining)))
                updated_baseline.append(discrete_remaining)

            # Keep expected arrivals aligned with perturbed ship ETAs so
            # next solve uses the moved predictions.
            self.ship_origin_latent[size] = updated_latent
            self.ship_origin[size] = updated_origin
            self.ship_origin_baseline[size] = updated_baseline
            self.expected_arrivals[size] = list(updated_origin)

            self.ship_destination[size] = [v - 1 for v in self.ship_destination[size]]
            self.expected_destinations[size] = [
                v - 1 for v in self.expected_destinations[size] if v >= 0
            ]

    def set_results(self, results):
        self.results = results

    def ship_tracking_snapshot(self):
        """Return compact diagnostics for ship accounting state."""
        per_ship = {}
        for size in self._fast_data["sets"]["ships"]:
            origin = self.ship_origin[size]
            expected = self.expected_arrivals[size]
            destination = self.ship_destination[size]
            expected_dest = self.expected_destinations[size]

            per_ship[size] = {
                "origin_len": len(origin),
                "expected_len": len(expected),
                "origin_expected_len_match": len(origin) == len(expected),
                "latent_len": len(self.ship_origin_latent[size]),
                "baseline_len": len(self.ship_origin_baseline[size]),
                "origin_due_today_count": sum(1 for v in origin if v <= 0),
                "expected_due_today_count": sum(1 for v in expected if v <= 0),
                "expected_due_next_day_count": sum(1 for v in expected if v == 1),
                "origin_min_days": int(min(origin)) if origin else None,
                "origin_max_days": int(max(origin)) if origin else None,
                "expected_min_days": int(min(expected)) if expected else None,
                "expected_max_days": int(max(expected)) if expected else None,
                "latent_min_days": float(min(self.ship_origin_latent[size])) if self.ship_origin_latent[size] else None,
                "latent_max_days": float(max(self.ship_origin_latent[size])) if self.ship_origin_latent[size] else None,
                "baseline_min_days": int(min(self.ship_origin_baseline[size])) if self.ship_origin_baseline[size] else None,
                "baseline_max_days": int(max(self.ship_origin_baseline[size])) if self.ship_origin_baseline[size] else None,
                "destination_len": len(destination),
                "expected_destination_len": len(expected_dest),
            }

        return {
            "iter_count": self.iter_count,
            "idx_hour": self.idx,
            "ship_tracking": per_ship,
        }
    
class PriceDynamics:
    def __init__(
        self,
        volatility,         # annual sigma
        long_run_log_price, # long-run mean of log-price
        speed,              # mean reversion speed (kappa)
        jump_intensity,     # annual lambda
        jump_mu,            # mean jump (log space)
        jump_sigma,         # jump volatility (log space)
        initial_price,
        seed=None
    ):
        self.sigma = volatility
        self.mu = long_run_log_price
        self.kappa = speed
        self.lambda_ = jump_intensity

        self.jump_mu = jump_mu
        self.jump_sigma = jump_sigma

        self.price = initial_price

        if seed is not None:
            np.random.seed(seed)

    def step(self, dt=1/365):
        """
        Exact OU discretisation + jump diffusion
        """
        X = np.log(self.price)

        # Precompute constants
        exp_kdt = np.exp(-self.kappa * dt)

        # Mean term
        mean = X * exp_kdt + self.mu * (1 - exp_kdt)

        # Variance term (exact)
        var = (self.sigma**2) * (1 - np.exp(-2 * self.kappa * dt)) / (2 * self.kappa)
        std = np.sqrt(var)

        # Gaussian shock
        diffusion = std * np.random.normal()

        # Jump component
        jump = self._jump_step(dt)

        # Update log-price
        X_new = mean + diffusion + jump

        # Back to price
        self.price = np.exp(X_new)

        return self.price

    def _jump_step(self, dt):
        """
        Compound Poisson jump in log space
        """
        if np.random.rand() < self.lambda_ * dt:
            return np.random.normal(self.jump_mu, self.jump_sigma)
        return 0.0