"""
MPC Controller Core Module
"""

from __future__ import annotations

import logging
from typing import Optional

from pyomo.environ import (
    AbstractModel,
    Any,
    Constraint,
    Objective,
    Param,
    Reals,
    Set,
    SolverFactory,
    Var,
    value,
)
from pyomo.opt import TerminationCondition

from algs.utils import ext_visualise_output, suppress_output


class MPCSolveError(RuntimeError):
    def __init__(self, termination_condition, message=None):
        self.termination_condition = termination_condition
        self.message = message
        super().__init__(
            f"MPC solve failed: {termination_condition}{' - ' + message if message else ''}"
        )


def find_equivalent_set(model, values):
    values_set = set(values)
    for component in model.component_objects(Set, active=True):
        if set(component.data()) == values_set:
            return component
    return None


class MPCController:
    def __init__(self):
        self.model1 = AbstractModel()
        self.model2 = AbstractModel()
        self._update_keys = None
        self._fig = None
        self._axs = None
        self._run_count = 0
        self.instance1 = None
        self.instance2 = None
        self.solver = None
        self._solver_configured = False
        self._solution_cache = {}
        self._cache_enabled = True

    # --- Public Interface Methods --- #

    def render(self):
        if self._fig is None:
            import matplotlib.pyplot as plt

            self._fig, self._axs = plt.subplots(4, 1, figsize=(14.0, 7.0), sharex=True)
        return self._fig

    def build(self, data: dict):
        """
        Docstring for build

        :param self: MPCController instance
        :param data: Data dictionary containing model definitions
        :type data: dict
        """
        self._build_sets(data["sets"])
        self._build_params(data["params"])
        self._build_vars(data["vars"])
        self._build_constraints(data["constraints"], data["forms"])
        self._build_equations(data["equations"], data["forms"])
        self._build_objectives(data["objectives"], data["forms"])
        self._init_fixed_flags()

    def solve(self, supress=False, solver="gurobi"):
        """
        This solves the MPCController model.

        :param self: MPCController instance
        :param supress: Whether to suppress solver output
        :param solver: Solver name
        """
        # Create model instances
        self.instance1 = self.model1.create_instance()
        self.instance2 = self.model2.create_instance()
        self._ensure_solver(solver)

        # Solve primary objective
        with suppress_output(supress):
            self.results = self.solver.solve(
                self.instance1, tee=supress, warmstart=False
            )

        # Set lexicographic flag
        self.lexicographic = 1

        # Lexicographic is just a re-solve of the same problem, there is a
        # functionality to set a secondary objective
        # ( see src/data/config.yaml -> formulations/secondary))
        if self.results.solver.termination_condition != TerminationCondition.optimal:
            if self._cache_enabled and self._solution_cache:
                self._apply_warm_start(self.instance2)
            self.results = self.solver.solve(
                self.instance2, tee=supress, warmstart=False
            )
            self.lexicographic = 2

        # If still not optimal after lexicographic fallback, raise so caller can handle/restart
        if self.results.solver.termination_condition != TerminationCondition.optimal:
            raise MPCSolveError(
                termination_condition=self.results.solver.termination_condition,
                message=getattr(self.results.solver, "message", None),
            )

        # Cache solution if enabled
        if (
            self._cache_enabled
            and self.results.solver.termination_condition == TerminationCondition.optimal
        ):
            self._cache_solution()

        return self.output()

    def update(self, stochastic_values, start_values: Optional[dict] = None):
        """
        Update the MPCController with new stochastic and starting values.

        :param self: MPCController instance
        :param stochastic_values: Stochastic values for update
        :param start_values: Starting values for update
        :type start_values: Optional[dict]
        """

        # If no start values, just do stochastic update
        if start_values is None:
            self.stochastic_update(data=stochastic_values)
            return None

        # Apply both start and stochastic updates
        self._apply_start_values(self.model1, start_values)
        self._apply_start_values(self.model2, start_values)
        self.stochastic_update(data=stochastic_values)

        return None

    def output(self, time_step: int = 24):
        """
        This grabs the conditional variables for the uncertainty.

        :param self: MPCController instance
        :param time_step: Time step for output
        :type time_step: int
        """

        # Lexicographic is just a flag to resolve the model
        solve = self.instance1 if self.lexicographic == 1 else self.instance2

        # Extract end states and outputs
        end_states = {}
        stochastic_output = {}
        stored_val = 0.0

        # Extract ships sent / ordered
        ordered_ship, sent_ship = self._ship_orders(solve, time_step)
        sent_vol_by_ship = self._sent_volumes(solve, time_step)

        # Setting these in outputs
        stochastic_output["ordered_ship"] = ordered_ship
        stochastic_output["sent_ship"] = sent_ship

        # Total sent volume (scalar)
        last_t = time_step
        hs, vs, cc, cv, hs0, vs0, cc0 = self._storage_terms(solve, last_t)

        # Calculate stored value
        stored_val += (hs - hs0) / 120.0
        stored_val += ((vs - vs0) * 1000.0 * cv) / 120.0
        stored_val += (cc - cc0) * cv / 120.0

        sent_vol = (sum(sent_vol_by_ship.values()) * cv) / 120

        end_states.update(
            self._end_states_at_time(solve, time_step=time_step, target_time=0)
        )

        # This fixes the value at the end of this solves control horizon to
        # be used as the start for the next solve
        getattr(self.model1, "fixed").set_value(True)
        getattr(self.model2, "fixed").set_value(True)

        return end_states, stochastic_output, stored_val, sent_vol

    def stochastic_update(self, data: Optional[dict] = None):
        """
        Docstring for stochastic_update

        :param self: MPCController instance
        :param data: Stochastic data for update
        :type data: Optional[dict]
        """

        # Iterate through each parameter and update its value
        for key, param in data.items():
            if param["param"]["initialize"] is not None:
                try:
                    self._update_param(key, param)
                except (AttributeError, KeyError):
                    if param["param"]["set"] is not None:
                        self._recreate_indexed_param(key, param)
                    else:
                        self._recreate_scalar_param(key, param)

                # These keys will be used to grab the output
                if param["loc"] == "endogenous":
                    if self._update_keys is None:
                        self._update_keys = {key: param["name"]}
                    elif key not in self._update_keys:
                        self._update_keys[key] = param["name"]

    # --- Internal Helper Methods --- #

    def _clean_variable_value(self, var, value):
        from pyomo.environ import Integers, NonNegativeIntegers, NonNegativeReals

        if value is None:
            return 0.0

        if var.domain in [NonNegativeIntegers, Integers]:
            cleaned_value = 0 if abs(value) < 1e-12 else int(round(value))
            if var.domain == NonNegativeIntegers and cleaned_value < 0:
                cleaned_value = 0
            return cleaned_value

        if var.domain == NonNegativeReals:
            if abs(value) < 1e-12:
                return 0.0
            if value < 0:
                if abs(value) > 1e-3:
                    logging.warning(
                        f"Variable {var.name} has negative value {value}, clamping to 0"
                    )
                return 0.0
            return value

        return value

    def _recreate_indexed_param(self, key, param):
        self.model1.del_component(key)
        self.model2.del_component(key)

        setattr(
            self.model1,
            key,
            Param(
                param["param"]["set"],
                initialize=param["param"]["initialize"],
                within=Reals,
                mutable=True,
            ),
        )
        getattr(self.model1, key).construct()
        setattr(
            self.model2,
            key,
            Param(
                param["param"]["set"],
                initialize=param["param"]["initialize"],
                within=Reals,
                mutable=True,
            ),
        )
        getattr(self.model2, key).construct()

    def _recreate_scalar_param(self, key, param):
        self.model1.del_component(key)
        self.model2.del_component(key)

        setattr(
            self.model1,
            key,
            Param(initialize=param["param"]["initialize"], within=Reals, mutable=True),
        )
        getattr(self.model1, key).construct()
        setattr(
            self.model2,
            key,
            Param(initialize=param["param"]["initialize"], within=Reals, mutable=True),
        )
        getattr(self.model2, key).construct()

    def visualise_output(self, time_step: int = 24):
        if self._fig is None:
            import matplotlib.pyplot as plt

            self._fig, self._axs = plt.subplots(4, 1, figsize=(14.0, 7.0), sharex=True)

        solve = self.instance1 if self.lexicographic == 1 else self.instance2

        self._joined_data = ext_visualise_output(
            solve=solve,
            axs=self._axs,
            run_count=self._run_count,
            time_step=time_step,
            joined_data=self._joined_data if hasattr(self, "_joined_data") else None,
            fig=self._fig,
        )

        self._run_count += 1

    def _cache_solution(self):
        active_instance = self.instance1 if self.lexicographic == 1 else self.instance2

        self._solution_cache.clear()
        for var in active_instance.component_objects(Var, active=True):
            for index in var:
                if var[index].value is not None:
                    cleaned_value = self._clean_variable_value(
                        var[index], var[index].value
                    )
                    self._solution_cache[(var.name, index)] = cleaned_value

    def _apply_warm_start(self, instance):
        try:
            for var in instance.component_objects(Var, active=True):
                for index in var:
                    cache_key = (var.name, index)
                    if cache_key in self._solution_cache:
                        cached_value = self._solution_cache[cache_key]
                        cleaned_value = self._clean_variable_value(
                            var[index], cached_value
                        )
                        var[index].set_value(cleaned_value)
        except Exception as e:
            logging.warning(f"Warm start failed: {e}")

    def clear_cache(self):
        self._solution_cache.clear()

    def set_cache_enabled(self, enabled: bool):
        self._cache_enabled = enabled
        if not enabled:
            self.clear_cache()

    def _end_states_at_time(self, solve, time_step, target_time):
        end_states = {}
        for var in solve.component_objects(Var):
            for index in var:
                if isinstance(index, tuple):
                    if index[-1] == time_step:
                        new_index = index[:-1] + (target_time,)
                        var_value = value(var[index])
                        var_value = self._clean_variable_value(var[index], var_value)
                        end_states[(var.name, new_index)] = var_value
                elif index == time_step:
                    var_value = value(var[index])
                    var_value = self._clean_variable_value(var[index], var_value)
                    end_states[(var.name, target_time)] = var_value
        return end_states

    def _clean_variable_value(self, var, value):
        from pyomo.environ import Integers, NonNegativeIntegers, NonNegativeReals

        if value is None:
            return 0.0

        if var.domain in [NonNegativeIntegers, Integers]:
            cleaned_value = 0 if abs(value) < 1e-12 else int(round(value))
            if var.domain == NonNegativeIntegers and cleaned_value < 0:
                cleaned_value = 0
            return cleaned_value

        if var.domain == NonNegativeReals:
            if abs(value) < 1e-12:
                return 0.0
            if value < 0:
                if abs(value) > 1e-3:
                    logging.warning(
                        f"Variable {var.name} has negative value {value}, clamping to 0"
                    )
                return 0.0
            return value

        return value

    def stochastic_update(self, data: Optional[dict] = None):
        for key, param in data.items():
            if param["param"]["initialize"] is not None:
                try:
                    if param["param"]["set"] is not None:
                        param1 = getattr(self.model1, key, None)
                        param2 = getattr(self.model2, key, None)

                        if param1 is not None and hasattr(param1, "set_values"):
                            param1.set_values(param["param"]["initialize"])
                        else:
                            self._recreate_indexed_param(key, param)

                        if param2 is not None and hasattr(param2, "set_values"):
                            param2.set_values(param["param"]["initialize"])
                    else:
                        param1 = getattr(self.model1, key, None)
                        param2 = getattr(self.model2, key, None)

                        if param1 is not None and hasattr(param1, "set_value"):
                            param1.set_value(param["param"]["initialize"])
                        else:
                            self._recreate_scalar_param(key, param)

                        if param2 is not None and hasattr(param2, "set_value"):
                            param2.set_value(param["param"]["initialize"])

                except (AttributeError, KeyError):
                    if param["param"]["set"] is not None:
                        self._recreate_indexed_param(key, param)
                    else:
                        self._recreate_scalar_param(key, param)

                # These keys will be used to grab the output
                if param["loc"] == "endogenous":
                    if self._update_keys is None:
                        self._update_keys = {key: param["name"]}
                    elif key not in self._update_keys:
                        self._update_keys[key] = param["name"]

    def _recreate_indexed_param(self, key, param):
        self.model1.del_component(key)
        self.model2.del_component(key)

        setattr(
            self.model1,
            key,
            Param(
                param["param"]["set"],
                initialize=param["param"]["initialize"],
                within=Reals,
                mutable=True,
            ),
        )
        getattr(self.model1, key).construct()
        setattr(
            self.model2,
            key,
            Param(
                param["param"]["set"],
                initialize=param["param"]["initialize"],
                within=Reals,
                mutable=True,
            ),
        )
        getattr(self.model2, key).construct()

    def _recreate_scalar_param(self, key, param):
        self.model1.del_component(key)
        self.model2.del_component(key)

        setattr(
            self.model1,
            key,
            Param(initialize=param["param"]["initialize"], within=Reals, mutable=True),
        )
        getattr(self.model1, key).construct()
        setattr(
            self.model2,
            key,
            Param(initialize=param["param"]["initialize"], within=Reals, mutable=True),
        )
        getattr(self.model2, key).construct()

    def visualise_output(self, time_step: int = 24):
        if self._fig is None:
            import matplotlib.pyplot as plt

            self._fig, self._axs = plt.subplots(4, 1, figsize=(14.0, 7.0), sharex=True)

        solve = self.instance1 if self.lexicographic == 1 else self.instance2

        self._joined_data = ext_visualise_output(
            solve=solve,
            axs=self._axs,
            run_count=self._run_count,
            time_step=time_step,
            joined_data=self._joined_data if hasattr(self, "_joined_data") else None,
            fig=self._fig,
        )

        self._run_count += 1

    def _cache_solution(self):
        active_instance = self.instance1 if self.lexicographic == 1 else self.instance2

        self._solution_cache.clear()
        for var in active_instance.component_objects(Var, active=True):
            for index in var:
                if var[index].value is not None:
                    cleaned_value = self._clean_variable_value(
                        var[index], var[index].value
                    )
                    self._solution_cache[(var.name, index)] = cleaned_value

    def _apply_warm_start(self, instance):
        try:
            for var in instance.component_objects(Var, active=True):
                for index in var:
                    cache_key = (var.name, index)
                    if cache_key in self._solution_cache:
                        cached_value = self._solution_cache[cache_key]
                        cleaned_value = self._clean_variable_value(
                            var[index], cached_value
                        )
                        var[index].set_value(cleaned_value)
        except Exception as e:
            logging.warning(f"Warm start failed: {e}")

    def _apply_start_values(self, model, start_values):
        for var in model.component_objects(Var, active=True):
            if var.name == "cumulative_profit":
                continue
            for index in var:
                key = (var.name, index)
                if key in start_values:
                    cleaned = self._clean_variable_value(var[index], start_values[key])
                    var[index].fix(cleaned)

    def _build_sets(self, sets_def):
        for key, set_ in sets_def.items():
            for model in (self.model1, self.model2):
                setattr(model, key, Set(initialize=list(set_)))
                getattr(model, key).construct()

    def _build_params(self, params_def):
        for key, param in params_def.items():
            if isinstance(param, dict):
                index_set_values = list(param.keys())
                for model in (self.model1, self.model2):
                    index_set = find_equivalent_set(model, index_set_values)
                    if index_set is None:
                        set_name = f"{key}_index"
                        index_set = Set(initialize=index_set_values, ordered=True)
                        setattr(model, set_name, index_set)
                    if not hasattr(model, key):
                        setattr(
                            model, key, Param(index_set, initialize=param, within=Any)
                        )
                        getattr(model, key).construct()
            else:
                for model in (self.model1, self.model2):
                    setattr(model, key, Param(initialize=param, within=Any))
                    getattr(model, key).construct()

    def _build_vars(self, vars_def):
        for key, var in vars_def.items():
            for model in (self.model1, self.model2):
                setattr(model, key, Var(*var["time_duration"], within=var["domain"]))
                getattr(model, key).construct()

    def _build_constraints(self, constraints_def, forms):
        for key, constraint in constraints_def.items():
            if key in forms["primary"]:
                setattr(
                    self.model1,
                    key,
                    Constraint(*constraint["time_duration"], rule=constraint["rule"]),
                )
            if key in forms["secondary"]:
                setattr(
                    self.model2,
                    key,
                    Constraint(*constraint["time_duration"], rule=constraint["rule"]),
                )

    def _build_equations(self, equations_def, forms):
        for key, equation in equations_def.items():
            if key in forms["primary"]:
                setattr(
                    self.model1,
                    key,
                    Constraint(*equation["time_duration"], rule=equation["rule"]),
                )
            if key in forms["secondary"]:
                setattr(
                    self.model2,
                    key,
                    Constraint(*equation["time_duration"], rule=equation["rule"]),
                )

    def _build_objectives(self, objectives_def, forms):
        for key, objective in objectives_def.items():
            if key in forms["primary"]:
                setattr(
                    self.model1,
                    key,
                    Objective(expr=objective["rule"], sense=objective["sense"]),
                )
            if key in forms["secondary"]:
                setattr(
                    self.model2,
                    key,
                    Objective(expr=objective["rule"], sense=objective["sense"]),
                )

    def _init_fixed_flags(self):
        for model in (self.model1, self.model2):
            setattr(model, "fixed", Param(initialize=False, mutable=True))
            getattr(model, "fixed").construct()

    def _ensure_solver(self, solver):
        if self._solver_configured:
            return
        self.solver = SolverFactory(solver)
        opts = self.solver.options
        opts["mipgap"] = 0.05
        opts["FeasibilityTol"] = 1e-6
        opts["OptimalityTol"] = 1e-9
        opts["Presolve"] = 2
        opts["Cuts"] = -1
        opts["Heuristics"] = 0.1
        opts["TimeLimit"] = 300
        opts["Threads"] = 0
        opts["Method"] = 2
        self._solver_configured = True

    def _update_param(self, key, param):
        if param["param"]["set"] is not None:
            param1 = getattr(self.model1, key, None)
            param2 = getattr(self.model2, key, None)
            if param1 is not None and hasattr(param1, "set_values"):
                param1.set_values(param["param"]["initialize"])
            else:
                self._recreate_indexed_param(key, param)

            if param2 is not None and hasattr(param2, "set_values"):
                param2.set_values(param["param"]["initialize"])
        else:
            param1 = getattr(self.model1, key, None)
            param2 = getattr(self.model2, key, None)
            if param1 is not None and hasattr(param1, "set_value"):
                param1.set_value(param["param"]["initialize"])
            else:
                self._recreate_scalar_param(key, param)

            if param2 is not None and hasattr(param2, "set_value"):
                param2.set_value(param["param"]["initialize"])

    def _ship_orders(self, solve, time_step):
        ordered_ship = {
            s: sum(
                value(getattr(solve, "n_ship_ordered")[s, t])
                for t in range(0, time_step, 24)
            )
            for s in getattr(solve, "ships", [])
        }
        sent_ship = {
            s: sum(
                value(getattr(solve, "n_ship_sent")[s, t])
                for t in range(0, time_step, 24)
            )
            for s in getattr(solve, "ships", [])
        }
        return ordered_ship, sent_ship

    def _sent_volumes(self, solve, time_step):
        return {
            s: sum(
                value(getattr(solve, "n_ship_sent")[s, t])
                for t in range(0, time_step, 24)
            )
            * value(getattr(solve, "ship_capacity")[s])
            for s in getattr(solve, "ships", [])
        }

    def _storage_terms(self, solve, last_t):
        hs = value(getattr(solve, "hydrogen_storage")[last_t])
        vs = value(getattr(solve, "vector_storage")[last_t])
        cc = value(getattr(solve, "cumulative_charge")[last_t])
        cv = value(getattr(solve, "calorific_value"))
        hs0 = value(getattr(solve, "hydrogen_storage")[0])
        vs0 = value(getattr(solve, "vector_storage")[0])
        cc0 = value(getattr(solve, "cumulative_charge")[0])
        return hs, vs, cc, cv, hs0, vs0, cc0
