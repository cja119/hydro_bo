"""
Dynamics for the contract layers: the site production problem (weather forecast only)
and the logistics problem (ships, buyer inventory and site deliveries).
"""

from math import exp, floor

from hydro_bo.envs.shipping.dynamics import BluePriceDynamics, Dynamics
from hydro_bo.envs.shipping.utils import generate_weather_forecast
from hydro_bo.utils.seeding import derive_subseed, make_rng


class SiteDynamics(Dynamics):
    def __init__(self, fast_data, args):
        self._fast_data = fast_data
        self._call_count = 0
        self._args = args

        seed = int(args["seed"])
        self._init_weather_data(derive_subseed(seed, "weather"))
        self._weather_data = self._weather_data * 2  # one weather year, repeated for runs longer than a year

        self.iter_count = 0
        self.idx = 0
        self.results = None

    def get_mpc_args(self):
        """Weather forecast for the next solve."""
        weather_forecast = generate_weather_forecast(
            self._weather_data,
            self.idx,
            self._fast_data["sets"]["grid0"],
            forecast_horizon=self._fast_data.get("forecast_horizon", 168),
            mean_override=self._args.get("weather_data", {}).get(
                "forecast_mean_override"
            ),
        )

        return {
            "energy_wind": {
                "name": "energy_wind",
                "loc": "exogenous",
                "param": {
                    "set": self._fast_data["sets"]["grid0"],
                    "initialize": weather_forecast,
                },
            },
        }

    def simulate_site_dynamics(self):
        """Advance the weather index by one day."""
        self.idx += 24
        self.iter_count += 1


class LogisticsDynamics(Dynamics):
    def __init__(self, fast_data, args, delivered, price):
        self._fast_data = fast_data
        self._call_count = 0
        self._args = args
        self._delivered = list(delivered)  # t vector per day
        self._mean_delivered = sum(self._delivered) / len(self._delivered)
        self._price = price  # $/kg

        p = fast_data["params"]
        self._arrival_time = int(round(p["mean_ship_arrival_time"]))
        self._arrival_std = float(p["std_ship_arrival_time"])
        self._transit_time = int(p["laden_transit_time"])
        self._reconversion_efficiency = float(p["reconversion_efficiency"])
        self._reconversion_cost = float(p["reconversion_cost"])  # $/kg H2-equivalent
        self._rng = make_rng(derive_subseed(int(args["seed"]), "dynamics"))

        pd_cfg = fast_data.get("price_dynamics", {})
        self._price_dynamics = None
        if pd_cfg.get("enabled", False) and "blue" in pd_cfg:
            self._price_dynamics = BluePriceDynamics(
                volatility=float(pd_cfg["sigma"]),
                speed=float(pd_cfg["kappa"]),
                jump_intensity=float(pd_cfg["lambda"]),
                jump_mu=float(pd_cfg["jump_mu"]),
                jump_sigma=float(pd_cfg["jump_sigma"]),
                seed=derive_subseed(int(args["seed"]), "price"),
                blue=pd_cfg["blue"],
                premium=float(pd_cfg.get("premium", 0.0)),
                penalty=float(pd_cfg.get("penalty", 9.0)),
            )

        self.iter_count = 0
        self.results = None
        self.ships = []

    def add_initial_orders(self, orders):
        """Ships ordered at t=0 of the first solve, placed the day before the run."""
        for size, count in orders.items():
            self._order_ships(size, count, -1, self._arrival_time - 1)

    def get_mpc_args(self):
        """Site deliveries, prices and ship arrivals for the next solve."""
        sets = self._fast_data["sets"]
        day = self.iter_count

        delivered = [self._delivered_on(day + t // 24 - 1) for t in sets["grid0"]]
        if self._price_dynamics is None:
            spot_forecast = fundamental_price = [self._price for _ in sets["grid0"]]
        else:
            # Netback fundamental price, today's realised spot price and the capped conditional mean spot price for each day ahead
            prices = self._price_dynamics
            ahead = [max(t // 24 - 1, 0) / 365 for t in sets["grid0"]]  # years
            fundamental_price = [self._netback(prices.floor_price(prices.t + h)) for h in ahead]
            spot_forecast = [self._netback(prices.price if h == 0 else prices.capped_mean(h)) for h in ahead]

        # Ships at origin: realised arrivals now, expected arrivals at their current ETA
        ship_arrived = {s: 0 for s in sets["ships"]}
        expected_ships = {(s, t): 0 for s in sets["ships"] for t in sets["grid1"]}
        laden_arrivals = {(s, t): 0 for s in sets["ships"] for t in sets["grid1"]}
        for ship in self.ships:
            if ship["origin_arrival"] is None and ship["eta"] < 1:
                ship["origin_arrival"] = day
            if ship["origin_arrival"] == day:
                ship_arrived[ship["size"]] += 1
            elif ship["origin_arrival"] is None:
                t = (floor(ship["eta"]) + 1) * 24
                if (ship["size"], t) in expected_ships:
                    expected_ships[ship["size"], t] += 1
            if ship["sent"] is not None and ship["sent"] <= day - 2 and ship["buyer_arrival"] >= day:
                t = (ship["buyer_arrival"] - day + 1) * 24
                if (ship["size"], t) in laden_arrivals:
                    laden_arrivals[ship["size"], t] += 1

        horizon_values = {}
        if int(self._fast_data["params"].get("terminal_valuation", 0)) in (2, 3) and self._price_dynamics is not None:
            # Storage-netback value of stock held at the start (day 0) and end (day T) of the horizon
            last = self.results or {}
            origin = last.get(("vector_storage", 0), 0.0) * float(self._fast_data["params"]["calorific_value"]) / 120
            buyer = last.get(("buyer_inventory", 0), 0.0)
            T = max(sets["grid0"]) // 24 - 1
            for prefix, days in (("tv0", 0), ("tv", T)):
                for key, val in self.storage_netback(days, origin, buyer).items():
                    name = f"{prefix}_{key}"
                    horizon_values[name] = {
                        "name": name,
                        "loc": "exogenous",
                        "param": {"set": None if not isinstance(val, dict) else list(val), "initialize": val},
                    }
            self.horizon_values = {k: v["param"]["initialize"] for k, v in horizon_values.items()}

        return {
            **horizon_values,
            "vector_delivered": {
                "name": "vector_delivered",
                "loc": "exogenous",
                "param": {"set": sets["grid0"], "initialize": delivered},
            },
            "spot_forecast": {
                "name": "spot_forecast",
                "loc": "exogenous",
                "param": {"set": sets["grid0"], "initialize": spot_forecast},
            },
            "fundamental_price": {
                "name": "fundamental_price",
                "loc": "exogenous",
                "param": {"set": sets["grid0"], "initialize": fundamental_price},
            },
            "ship_arrived": {
                "name": "ship_arrived",
                "loc": "exogenous",
                "param": {"set": sets["ships"], "initialize": ship_arrived},
            },
            "expected_ships": {
                "name": "expected_ships",
                "loc": "exogenous",
                "param": {"set": (sets["ships"], sets["grid1"]), "initialize": expected_ships},
            },
            "laden_arrivals": {
                "name": "laden_arrivals",
                "loc": "exogenous",
                "param": {"set": (sets["ships"], sets["grid1"]), "initialize": laden_arrivals},
            },
        }

    def spot_price(self):
        """Realised netback spot price today [$/kg] and whether the premium cap binds on the hub price."""
        if self._price_dynamics is None:
            return self._price, False
        return self._netback(self._price_dynamics.price), bool(self._price_dynamics.cap_binds())

    def storage_netback(self, days, origin, buyer):
        """Storage netback of stock held `days` ahead [$/kg H2]: the forecast netback on the expected sale date at the bid,
        less the costs still to pay to reach the buyer, discounted at the project rate to today.
        terminal_valuation 2: bid (1 - eps). terminal_valuation 3: marginal bid (1 - eps (1 + 2 q_spot / q_ref)), the
        half-spread of selling one extra kg on top of the day's spot volume q_spot = (1 - f) X/30 (spot_daily_volume), each
        location's value floored at 0 per kg (the discharge charge of a ship at sea is netted per kg landed before the floor).
        `origin` and `buyer` are today's stock [kt H2]; they set the sell-down time at q_ref (held fixed over the horizon).
        Returns origin (per kg in the tank), buyer (per kg in the tank), and per ship size and remaining laden days k:
        sea (per kg landed) and sea_disc (discount on the per-ship discharge charge; 0 under mode 3, where it is netted)."""
        p = self._fast_data["params"]
        prices = self._price_dynamics
        r = float(p["discount_factor"])
        eps, q_ref = float(p["half_spread"]), float(p["reference_volume"])
        marginal = int(p.get("terminal_valuation", 2)) == 3
        spread = eps * (1 + 2 * float(p.get("spot_daily_volume", 0.0)) / q_ref) if marginal else eps
        floor = (lambda x: max(x, 0.0)) if marginal else (lambda x: x)
        h2 = float(p["calorific_value"]) / 120
        laden, ballast, port = int(p["laden_transit_time"]), float(p["ballast_transit_time"]), float(p["port_time"])

        def disc(d):
            return exp(-r * d / 365)

        def bid(d):
            return (1 - spread) * self._netback(prices.price if d <= 0 else prices.capped_mean(d / 365))

        # Origin tank: fill one large cargo, load, sail laden, discharge, sell down; the large ship pays the round trip
        cap = float(p["ship_capacity"]["large"])
        cargo = cap * h2 / 1000                                                           # kt H2
        fuel_share = float(p["ship_fuel_rate"]["large"]) * (laden + ballast) / cap
        round_trip = (float(p["ship_charter_rate"]["large"]) * (float(p["mean_ship_arrival_time"]) + laden + ballast + port) * 24
                      + 2 * float(p["port_fee"]["large"])) / 1_000_000 / cargo           # $/kg H2
        forecast = getattr(self, "delivery_forecast", None) or self._delivered            # t carrier per day
        rate = sum(forecast) / len(forecast) * h2 / 1000                                   # kt H2 per day
        fill = cargo / max(rate, 1e-9)                                                     # days
        d_origin = days + fill + laden + port + origin / q_ref
        values = {"origin": floor(disc(d_origin) * ((1 - fuel_share) * bid(d_origin) - round_trip))}

        # At sea: remaining laden days k, then discharge and sell down one cargo
        sea, sea_disc = {}, {}
        for s in self._fast_data["sets"]["ships"]:
            landed = (float(p["ship_capacity"][s]) - float(p["ship_fuel_rate"][s]) * (laden + ballast)) * h2 / 1000
            charge = (float(p["ship_charter_rate"][s]) * (ballast + port) * 24 + float(p["port_fee"][s])) / 1_000_000 / landed
            for k in range(laden + 1):
                d = days + k + landed / q_ref
                if marginal:
                    sea[f"{s}:{k}"] = floor(disc(d) * (bid(d) - charge))
                    sea_disc[f"{s}:{k}"] = 0.0
                else:
                    sea[f"{s}:{k}"] = disc(d) * bid(d)
                    sea_disc[f"{s}:{k}"] = disc(d)
        values.update({"sea": sea, "sea_disc": sea_disc})

        # Buyer tank: sell down only (import storage and reconversion are in the netback)
        d_buyer = days + buyer / q_ref
        values["buyer"] = floor(disc(d_buyer) * bid(d_buyer))
        return values

    def _netback(self, price):
        """Carrier price at the export terminal from a hub hydrogen price [$/kg H2-equivalent]."""
        return self._reconversion_efficiency * price - self._reconversion_cost

    def simulate_logistics_dynamics(self):
        """Apply the first day of the last solve: orders, sends and discharges."""
        day = self.iter_count
        for size in self._fast_data["sets"]["ships"]:
            n_ordered = round(self.results[("n_ship_ordered", (size, 0))])
            n_sent = round(self.results[("n_ship_sent", (size, 0))])
            n_discharged = round(self.results[("n_ship_discharged", (size, 0))])

            self._order_ships(size, n_ordered, day, self._arrival_time)

            # Sends and discharges take the longest-waiting ships first
            waiting = [
                s for s in self.ships
                if s["size"] == size and s["origin_arrival"] is not None
                and s["origin_arrival"] <= day and s["sent"] is None
            ]
            for ship in sorted(waiting, key=lambda s: s["origin_arrival"])[:n_sent]:
                ship["sent"] = day
                ship["buyer_arrival"] = day + self._transit_time

            holding = [
                s for s in self.ships
                if s["size"] == size and s["buyer_arrival"] is not None
                and s["buyer_arrival"] <= day and s["discharged"] is None
            ]
            for ship in sorted(holding, key=lambda s: s["buyer_arrival"])[:n_discharged]:
                ship["discharged"] = day

        # Ships still on their way to origin: one day closer, with noise beyond the last day
        for ship in self.ships:
            if ship["origin_arrival"] is None:
                disturbance = 0.0
                if ship["eta"] > 1:
                    disturbance = float(self._rng.normal(0.0, self._arrival_std))
                ship["eta"] = ship["eta"] - 1 + disturbance

        if self._price_dynamics is not None:
            self._price_dynamics.step(dt=1 / 365)
        self.iter_count += 1

    def ship_states(self, day):
        """Ships waiting at origin, at sea and holding at the buyer at the end of `day`, by size."""
        states = {
            state: {s: 0 for s in self._fast_data["sets"]["ships"]}
            for state in ("waiting", "at_sea", "holding")
        }
        for ship in self.ships:
            if ship["origin_arrival"] is None:
                continue
            if ship["origin_arrival"] <= day and (ship["sent"] is None or ship["sent"] > day):
                states["waiting"][ship["size"]] += 1
            elif ship["sent"] is not None and ship["sent"] <= day < ship["buyer_arrival"]:
                states["at_sea"][ship["size"]] += 1
            elif (
                ship["buyer_arrival"] is not None and ship["buyer_arrival"] <= day
                and (ship["discharged"] is None or ship["discharged"] > day)
            ):
                states["holding"][ship["size"]] += 1
        return states

    def _order_ships(self, size, count, day, eta):
        for _ in range(count):
            self.ships.append({
                "size": size,
                "ordered": day,
                "eta": float(eta),  # days until arrival at origin
                "origin_arrival": None,
                "sent": None,
                "buyer_arrival": None,
                "discharged": None,
            })

    def _delivered_on(self, day):
        if day < 0:
            return 0.0
        if day >= len(self._delivered):
            return self._mean_delivered
        return self._delivered[day]


class ContractDynamics(LogisticsDynamics):
    def __init__(self, fast_data, args, price):
        super().__init__(fast_data, args, [0.0], price)
        self.delivery_forecast = None  # t vector per day over the logistics horizon

    def get_mpc_args(self):
        """Ship arrivals and prices as LogisticsDynamics, with the day's delivery forecast."""
        mpc_args = super().get_mpc_args()
        mpc_args["vector_delivered"]["param"]["initialize"] = list(self.delivery_forecast)
        return mpc_args
