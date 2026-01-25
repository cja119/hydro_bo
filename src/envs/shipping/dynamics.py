from numpy.random import normal
from pyomo.environ import value
from meteor_py import GetData
from .utils import temporal_align, generate_weather_forecast, count_and_shift_arrivals


class Dynamics:
    def __init__(self, fast_data, slow_data, args):
        self._fast_data = fast_data
        self._slow_data = slow_data
        self._call_count = 0
        self._args = args

        self._init_weather_data()

        self.new_states = None
        self.iter_count = 0
        self.idx = 0
        self.results = None

        self.states = self._init_state_variables()
        self.destination_storage = 0.5 * slow_data["params"]["storage_capacity"]

        self.ship_destination = {s: [] for s in fast_data["sets"]["ships"]}
        self.ship_origin = {s: [] for s in fast_data["sets"]["ships"]}
        self.expected_arrivals = {s: [] for s in fast_data["sets"]["ships"]}
        self.expected_destinations = {s: [] for s in fast_data["sets"]["ships"]}

    def _init_weather_data(self):
        weather_file = self._args["weather_data"]["weather_file"]
        self._weather_data = GetData([weather_file]).data()
        self._weather_data = temporal_align(self._weather_data, randomise=True)

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

        for size in self.ship_origin:
            origin_arrive[size] = count_and_shift_arrivals(
                self.ship_origin, self.expected_arrivals, size
            )
            for arrival in self.expected_arrivals[size]:
                expected_arrivals_indexed[size, int(arrival) * 24] += 1

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
                    "set": list(expected_arrivals_indexed.keys()),
                    "initialize": expected_arrivals_indexed,
                },
            },
        }

    def simulate_shipping_dynamics(self):
        self._simulate_origin_shipping()
        self._decrement_ship_counters()
        self.idx += 24
        self.iter_count += 1

    def _simulate_origin_shipping(self):
        p = self._fast_data["params"]
        for size, num_orders in self.states["ordered_ship"].items():
            if num_orders <= 0:
                continue

            arrivals = [
                int(
                    normal(
                        value(p["mean_ship_arrival_time"]),
                        value(p["std_ship_arrival_time"]),
                    )
                )
                for _ in range(int(num_orders))
            ]
            self.ship_origin[size].extend(arrivals)
            self.expected_arrivals[size].extend(
                [p["mean_ship_arrival_time"]] * int(num_orders)
            )

    def _decrement_ship_counters(self):
        for size in self._fast_data["sets"]["ships"]:
            self.ship_origin[size] = [v - 1 for v in self.ship_origin[size]]
            self.expected_arrivals[size] = [
                v - 1 for v in self.expected_arrivals[size] if v >= 1
            ]
            self.ship_destination[size] = [v - 1 for v in self.ship_destination[size]]
            self.expected_destinations[size] = [
                v - 1 for v in self.expected_destinations[size] if v >= 0
            ]

    def set_results(self, results):
        self.results = results
