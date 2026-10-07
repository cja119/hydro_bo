from pyomo.environ import value
from meteor_py import GetData
from math import floor
from .utils import temporal_align, generate_weather_forecast, count_and_shift_arrivals
from hydro_bo.utils.seeding import derive_subseed, make_rng

import numpy as np

class Dynamics:
    def __init__(self, fast_data, slow_data, args):
        self._fast_data = fast_data
        self._slow_data = slow_data
        self._call_count = 0
        self._args = args

        seed = int(args["seed"])
        self._rng = make_rng(derive_subseed(seed, "dynamics"))

        self._init_weather_data(derive_subseed(seed, "weather"))

        self.new_states = None
        self.iter_count = 0
        self.idx = 0
        self.results = None

        self.states = self._init_state_variables()
        self.destination_storage = 0.5 * slow_data["params"]["storage_capacity"]

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
                seed=derive_subseed(seed, "price"),
                fundamental=pd_cfg.get("fundamental"),
            )
            self.h2_price = float(self._price_dynamics.price)
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
        keys = list(weather_file) if isinstance(weather_file, (list, tuple)) else [weather_file]
        self._weather_data = GetData(keys).data()
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
            self._weather_data,
            self.idx,
            self._fast_data["sets"]["grid0"],
            forecast_horizon=self._fast_data.get("forecast_horizon", 168),
            mean_override=self._args.get("weather_data", {}).get(
                "forecast_mean_override"
            ),
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
                disturbance_days = 0.0 if float(remaining_days) <= 1.0 else float(self._rng.normal(0.0, std_days))
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
    """
    Log-price OU + compound-Poisson jump process.

    Without `fundamental`, the log-price reverts to `long_run_log_price`
    (original behaviour). With a `fundamental` dict (keys `h_g`, `P_NG`,
    `k_g`, `e_g`, `lambda_0`, `lambda_inf`, `kappa_lambda`, optional `t_0`),
    the price is P_f(t) * exp(y): P_f is the grey-hydrogen cost floor

        lambda(t) = lambda_inf + (lambda_0 - lambda_inf) * exp(-kappa_lambda * (t_0 + t))
        P_f(t)    = h_g * P_NG + k_g + e_g * lambda(t)        (t in years of simulation)

    and y follows the same OU + jump step, reverting to `theta`, chosen so
    that the stationary E[exp(y)] = 1. `initial_price` and
    `long_run_log_price` are unused in that mode; y starts at theta.
    """

    def __init__(
        self,
        volatility,         # annual sigma
        long_run_log_price, # long-run mean of log-price
        speed,              # mean reversion speed (kappa)
        jump_intensity,     # annual lambda
        jump_mu,            # mean jump (log space)
        jump_sigma,         # jump volatility (log space)
        initial_price,
        seed=None,
        fundamental=None,   # optional grey-hydrogen floor parameters
    ):
        self.sigma = volatility
        self.mu = long_run_log_price
        self.kappa = speed
        self.lambda_ = jump_intensity

        self.jump_mu = jump_mu
        self.jump_sigma = jump_sigma

        self._rng = make_rng(seed) if seed is not None else make_rng(0)

        self.fundamental = None if fundamental is None else {
            **{
                k: float(fundamental[k])
                for k in ("h_g", "P_NG", "k_g", "e_g", "lambda_0", "lambda_inf", "kappa_lambda")
            },
            "t_0": float(fundamental.get("t_0", 0.0)),  # years from today to the start of the simulation
        }
        if self.fundamental is None:
            self.price = initial_price
        else:
            self.t = 0.0
            self.theta = self.compensated_theta()
            self.y = self.theta
            self.price = self.floor_price(self.t) * np.exp(self.y)

    def compensated_theta(self):
        """theta = -sigma^2/(4 kappa)
                   - (lambda/kappa) * int_0^1 (exp(jump_mu*v + jump_sigma^2*v^2/2) - 1) / v dv,
        which sets the stationary E[exp(y)] of the continuous-time process to 1."""
        from scipy.integrate import quad

        integral, _ = quad(
            lambda v: np.expm1(self.jump_mu * v + 0.5 * self.jump_sigma**2 * v**2) / v,
            0.0, 1.0,
        )
        return -self.sigma**2 / (4 * self.kappa) - (self.lambda_ / self.kappa) * integral

    def conditional_mean(self, h):
        """E[S(t+h) | y_t] for horizon `h` years from the current state:

            P_f(t+h) * exp(m + v + j)
            m = theta + (y_t - theta) * exp(-kappa h)
            v = sigma^2 (1 - exp(-2 kappa h)) / (4 kappa)
            j = (lambda/kappa) * int_{exp(-kappa h)}^1
                    (exp(jump_mu u + jump_sigma^2 u^2 / 2) - 1) / u du

        Continuous-time result; fundamental mode only."""
        if self.fundamental is None:
            raise ValueError("conditional_mean needs the fundamental price block")
        from scipy.integrate import quad

        decay = np.exp(-self.kappa * h)
        m = self.theta + (self.y - self.theta) * decay
        v = self.sigma**2 * (1 - np.exp(-2 * self.kappa * h)) / (4 * self.kappa)
        integral, _ = quad(
            lambda u: np.expm1(self.jump_mu * u + 0.5 * self.jump_sigma**2 * u**2) / u,
            decay, 1.0,
        )
        j = (self.lambda_ / self.kappa) * integral
        return self.floor_price(self.t + h) * np.exp(m + v + j)

    def carbon_price(self, t):
        f = self.fundamental
        return f["lambda_inf"] + (f["lambda_0"] - f["lambda_inf"]) * np.exp(-f["kappa_lambda"] * (f["t_0"] + t))

    def floor_price(self, t):
        f = self.fundamental
        return f["h_g"] * f["P_NG"] + f["k_g"] + f["e_g"] * self.carbon_price(t)

    def step(self, dt=1/365):
        """
        Exact OU discretisation + jump diffusion
        """
        if self.fundamental is not None:
            return self._fundamental_step(dt)

        X = np.log(self.price)

        # Precompute constants
        exp_kdt = np.exp(-self.kappa * dt)

        # Mean term
        mean = X * exp_kdt + self.mu * (1 - exp_kdt)

        # Variance term (exact)
        var = (self.sigma**2) * (1 - np.exp(-2 * self.kappa * dt)) / (2 * self.kappa)
        std = np.sqrt(var)

        # Gaussian shock
        diffusion = std * self._rng.normal()

        # Jump component
        jump = self._jump_step(dt)

        # Update log-price
        X_new = mean + diffusion + jump

        # Back to price
        self.price = np.exp(X_new)

        return self.price

    def _fundamental_step(self, dt):
        """Same OU + jump step as `step`, on y with long-run level theta;
        price = P_f(t) * exp(y)."""
        exp_kdt = np.exp(-self.kappa * dt)
        mean = self.y * exp_kdt + self.theta * (1 - exp_kdt)
        var = (self.sigma**2) * (1 - np.exp(-2 * self.kappa * dt)) / (2 * self.kappa)
        diffusion = np.sqrt(var) * self._rng.normal()
        jump = self._jump_step(dt)

        self.y = mean + diffusion + jump
        self.t += dt
        self.price = self.floor_price(self.t) * np.exp(self.y)

        return self.price

    def _jump_step(self, dt):
        """
        Compound Poisson jump in log space
        """
        if self._rng.random() < self.lambda_ * dt:
            return float(self._rng.normal(self.jump_mu, self.jump_sigma))
        return 0.0


class BluePriceDynamics(PriceDynamics):
    """
    Green-hydrogen price anchored on the blue-hydrogen cost, on the basis y:

        P_b = h * P_NG + k + e * lambda + q * c_TS
        P_g = P_b + premium
        S_g = P_b * exp(y) + min(premium * exp(y), penalty)

    `blue` holds `h`, `k`, `e`, `q`, `c_TS`, `P_NG` and `lambda`; y follows the
    fundamental-mode OU + jump step with the compensated theta. `conditional_mean`
    is that of P_g * exp(y), without the cap.
    """

    def __init__(
        self,
        volatility,         # annual sigma
        speed,              # mean reversion speed (kappa)
        jump_intensity,     # annual lambda
        jump_mu,            # mean jump (log space)
        jump_sigma,         # jump volatility (log space)
        seed=None,
        blue=None,          # blue-hydrogen cost parameters
        premium=0.0,        # green premium over blue ($/kg)
        penalty=9.0,        # cap on the premium in the spot price ($/kg)
    ):
        super().__init__(volatility, None, speed, jump_intensity, jump_mu, jump_sigma, None, seed=seed)
        self.fundamental = {
            **{k: float(blue[k]) for k in ("h", "k", "e", "q", "c_TS", "P_NG", "lambda")},
            "premium": float(premium),
            "penalty": float(penalty),
        }
        self.t = 0.0
        self.theta = self.compensated_theta()
        self.y = self.theta
        self.price = self.green_spot()

    def blue_price(self, t):
        f = self.fundamental
        return f["h"] * f["P_NG"] + f["k"] + f["e"] * f["lambda"] + f["q"] * f["c_TS"]

    def floor_price(self, t):
        return self.blue_price(t) + self.fundamental["premium"]

    def cap_binds(self):
        return self.fundamental["premium"] * np.exp(self.y) > self.fundamental["penalty"]

    def capped_mean(self, h):
        """E[S_g(t+h) | y_t]: the uncapped mean less the Black call value of the cap, with y(t+h) Gaussian."""
        from scipy.stats import norm

        f = self.fundamental
        mean = self.conditional_mean(h)
        if f["premium"] == 0:
            return mean
        F = mean / self.floor_price(self.t + h)
        s = np.sqrt(
            (self.sigma**2 + self.lambda_ * (self.jump_mu**2 + self.jump_sigma**2))
            * (1 - np.exp(-2 * self.kappa * h)) / (2 * self.kappa)
        )
        d1 = (np.log(f["premium"] * F / f["penalty"]) + s**2 / 2) / s
        d2 = d1 - s
        return mean - (f["premium"] * F * norm.cdf(d1) - f["penalty"] * norm.cdf(d2))

    def green_spot(self):
        f = self.fundamental
        return self.blue_price(self.t) * np.exp(self.y) + min(f["premium"] * np.exp(self.y), f["penalty"])

    def step(self, dt=1/365):
        super().step(dt)
        self.price = self.green_spot()
        return self.price
