"""
Contract environment: the site production controller and the logistics controller,
solved one after the other each day.
"""

from __future__ import annotations

import time
from pathlib import Path

from pyomo.environ import value

from hydro_bo.envs.contract.dynamics import ContractDynamics, SiteDynamics
from hydro_bo.envs.shipping.utils import args_dict, import_mpc_data, import_mpc_functions
from hydro_bo.mpc.mpc import LogisticsMPCController, SiteMPCController
from hydro_bo.utils.logging_config import get_logger
from hydro_bo.utils.seeding import derive_subseed


logger = get_logger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SITE_DATA_DIR = DATA_DIR / "site"
LOGISTICS_DATA_DIR = DATA_DIR / "logistics"
SITE_EQUATIONS_PATH = Path(__file__).parent / "site_equations.py"
LOGISTICS_EQUATIONS_PATH = Path(__file__).parent / "logistics_equations.py"
PLAN_DAYS = 7                                                    # days of the site plan in the forecast
EVALUATED_DAYS = 365                                             # days evaluated after the warm-up
DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # as Dynamics.get_forecasts


class ContractEnv:

    def __init__(self) -> None:
        self._args = args_dict()
        self._args["contract"] = {
            "logistics_settings": {},  # parameter overrides for the logistics model
            "price_settings": {},      # price_dynamics overrides for the logistics model
            "monthly_delivery": None,  # mean daily delivery per env month [t vector]
            "price": 5.0,              # $/kg
        }
        self._site = SiteMPCController()
        self._logistics = LogisticsMPCController()
        self.records = []

    def __enter__(self) -> None:
        return self._args

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._build_env()

    def _build_env(self) -> None:
        seed = int(self._args["seed"])
        params = self._args["mpc"]["planning_model"]
        contract = self._args["contract"]

        self._site = SiteMPCController(gurobi_seed=derive_subseed(seed, "gurobi"))
        self._site_data = import_mpc_data(
            dict(params),
            self._args["vector"],
            self._args["mpc"]["random_param"],
            random_param_seed=derive_subseed(seed, "random_param"),
            data_folder=SITE_DATA_DIR,
        )
        self._site_data.update(
            import_mpc_functions(
                SITE_DATA_DIR,
                self._site_data["sets"],
                equations_path=SITE_EQUATIONS_PATH,
            )
        )
        self._site.build(self._site_data)
        self._site_dynamics = SiteDynamics(self._site_data, self._args)

        self._logistics = LogisticsMPCController(gurobi_seed=derive_subseed(seed, "gurobi"))
        self._logistics_data = import_mpc_data(
            dict(params),
            self._args["vector"],
            data_folder=LOGISTICS_DATA_DIR,
        )
        self._logistics_data.update(
            import_mpc_functions(
                LOGISTICS_DATA_DIR,
                self._logistics_data["sets"],
                equations_path=LOGISTICS_EQUATIONS_PATH,
            )
        )
        self._logistics_data["params"].update(contract["logistics_settings"])
        self._logistics_data["price_dynamics"].update(contract["price_settings"])
        self._logistics.build(self._logistics_data)
        self._dynamics = ContractDynamics(self._logistics_data, self._args, contract["price"])
        self._warm_up_days = self._logistics_data["contract"].get("warm_up_days", 30)
        self._n_days = self._warm_up_days + EVALUATED_DAYS

        self._previous_delivered = 0.0
        self.records = []

    def reset(self) -> None:
        self._build_env()

    def step(self, action):
        n_steps = self._site_dynamics.get_forecasts()
        self._dynamics.get_forecasts()

        reward = 0.0
        for _ in range(n_steps):
            if self._site_dynamics.iter_count >= self._n_days:
                break
            reward += self._step_day()

        call_count = self._site_dynamics._call_count
        done = self._site_dynamics.iter_count >= self._n_days
        return {"records": self.records}, reward, done, {"call_count": call_count}

    def _step_day(self):
        """Site solve, then logistics solve; apply day 0 of each. Returns the day's cash flow [$M]."""
        day = self._site_dynamics.iter_count
        month = self._site_dynamics._call_count
        solver = self._args["mpc"]["solver"]

        # Site production
        start = time.perf_counter()
        self._site.update(self._site_dynamics.get_mpc_args(), self._site_dynamics.results)
        site_end_states, delivered, planned_delivered = self._site.solve(solver=solver)
        site_solve_s = time.perf_counter() - start
        site_status = str(self._site.results.solver.termination_condition)
        self._site_dynamics.set_results(site_end_states)

        # Logistics, with the site's delivery forecast
        self._dynamics.delivery_forecast = self._delivery_forecast(day, delivered, planned_delivered)
        mpc_args = self._dynamics.get_mpc_args()
        start = time.perf_counter()
        self._logistics.update(mpc_args, self._dynamics.results)
        end_states = self._logistics.solve(solver=solver)
        logistics_solve_s = time.perf_counter() - start

        inst = self._logistics.instance
        if day == 0:
            self._dynamics.add_initial_orders({s: round(value(inst.n_ship_ordered[s, 0])) for s in inst.ships})
        gurobi_model = self._logistics.solver._solver_model

        spot_price, cap_binds = self._dynamics.spot_price()
        self._dynamics.set_results(end_states)
        self._dynamics.simulate_logistics_dynamics()
        self._site_dynamics.simulate_site_dynamics()
        self._previous_delivered = delivered

        record = self._record_day(inst, day, month, delivered)
        site = self._site.instance
        record.update({
            "site_hydrogen_produced": sum(value(site.hydrogen_produced[t]) for t in range(1, 25)),
            "site_energy_curtailed": sum(value(site.energy_curtailed[t]) for t in range(1, 25)),
            "site_hydrogen_storage": value(site.hydrogen_storage[24]),
            "spot_price": spot_price,                                     # $/kg, realised
            "cap_binds": cap_binds,
            "site_solve_s": site_solve_s,
            "site_status": site_status,
            "solve_s": logistics_solve_s,
            "status": str(self._logistics.results.solver.termination_condition),
            "relaxations": self._logistics._relaxation_stats["relaxations_per_solve"][-1],
            "obj_val": gurobi_model.ObjVal,                               # $M
            "bound_gap": gurobi_model.ObjBound - gurobi_model.ObjVal,     # $M
        })
        self.records.append(record)
        logger.info("contract_env.day", day=day, month=month, site_delivered_t=delivered,
                    site_solve_s=site_solve_s, logistics_solve_s=logistics_solve_s)
        return self._cash_flow(record)

    def _delivery_forecast(self, day, delivered, planned_delivered):
        """Delivered vector per logistics step [t]: today's actual, the site plan for days 1-7, then monthly means."""
        forecast = []
        for t in self._logistics_data["sets"]["grid0"]:
            k = t // 24 - 1                      # days after today
            if k < 0:
                forecast.append(self._previous_delivered)
            elif k == 0:
                forecast.append(delivered)
            elif k <= PLAN_DAYS and 24 * (k + 1) in planned_delivered:
                forecast.append(sum(planned_delivered[h] for h in range(24 * k + 1, 24 * (k + 1) + 1)))
            else:
                forecast.append(self._args["contract"]["monthly_delivery"][self._month_of(day + k) - 1])
        return forecast

    def _month_of(self, day):
        """Env month (1-12) of a day of the run, with months in the order Dynamics.get_forecasts uses."""
        lengths = [DAYS_IN_MONTH[m % 12] for m in range(1, 13)]
        day = day % sum(lengths)
        for month, length in enumerate(lengths, start=1):
            if day < length:
                return month
            day -= length

    def _record_day(self, inst, day, month, delivered):
        ships = list(inst.ships)
        upper = getattr(inst, "cons2")[24]
        storage_end = value(inst.vector_storage[24])
        states = self._dynamics.ship_states(day)
        return {
            "day": day,
            "month": month,
            "evaluated": day >= self._warm_up_days,
            "site_delivered_t": delivered,
            "delivered_t": value(inst.vector_delivered[24]),
            "delivery_forecast": [value(inst.vector_delivered[t]) for t in inst.grid0][1:],
            "storage_plan": [value(inst.vector_storage[t]) for t in inst.grid0],
            "storage_start": value(inst.vector_storage[0]),
            "storage_end": storage_end,
            "storage_limit": storage_end - value(upper.body),
            "flared": value(inst.vector_flared[24]),
            "ordered": [round(value(inst.n_ship_ordered[s, 24])) for s in ships],
            "sent": [round(value(inst.n_ship_sent[s, 24])) for s in ships],
            "discharged": [round(value(inst.n_ship_discharged[s, 24])) for s in ships],
            "waiting_model": [value(inst.waiting_ships[s, 24]) for s in ships],
            "holding_model": [value(inst.ships_holding[s, 24]) for s in ships],
            "waiting": [states["waiting"][s] for s in ships],
            "at_sea": [states["at_sea"][s] for s in ships],
            "holding": [states["holding"][s] for s in ships],
            "buyer_inventory_start": value(inst.buyer_inventory[0]),
            "buyer_inventory": value(inst.buyer_inventory[24]),
            "spot_bought": value(inst.spot_bought[24]),
            "spot_sold": value(inst.spot_sold[24]),
            "spot_price_model": value(inst.spot_forecast[24]),                          # $/kg, uncapped
            "fundamental": value(inst.fundamental_price[24]),                           # $/kg
            "spot_forecast_path": [value(inst.spot_forecast[t]) for t in inst.grid0][1:],
        }

    def _cash_flow(self, record):
        """Contract revenue at the fundamental price and spot trades at the realised price with linear impact,
        less voyage charter, port fees and holding loss [$M]."""
        p = self._logistics_data["params"]
        spot, eps, q_ref = record["spot_price"], p["half_spread"], p["reference_volume"]
        ships = list(self._logistics_data["sets"]["ships"])
        charter = sum(
            record["ordered"][j] * p["ship_charter_rate"][s] * p["mean_ship_arrival_time"] * 24
            + record["sent"][j] * (p["ship_charter_rate"][s] * p["laden_transit_time"] * 24 + p["port_fee"][s])
            + record["discharged"][j]
            * (p["ship_charter_rate"][s] * (p["ballast_transit_time"] + p["port_time"]) * 24 + p["port_fee"][s])
            + record["waiting_model"][j] * (p["ship_berthing_rate"][s] + p["ship_charter_rate"][s]) * 24
            + record["holding_model"][j] * p["ship_charter_rate"][s] * 24
            for j, s in enumerate(ships)
        ) / 1_000_000
        holding_loss = sum(
            p["holding_loss_rate"] * record["holding_model"][j] * p["ship_capacity"][s] * p["calorific_value"] / 120 / 1000 * spot
            for j, s in enumerate(ships)
        )
        return (
            record["fundamental"] * p["commitment"] / 30
            + (1 - eps * (1 + record["spot_sold"] / q_ref)) * spot * record["spot_sold"]
            - (1 + eps * (1 + record["spot_bought"] / q_ref)) * spot * record["spot_bought"]
            - charter
            - holding_loss
        )
