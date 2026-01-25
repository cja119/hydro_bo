"""
This defines the inner loop of the model predictive control (MPC) algorithm.
"""

from __future__ import annotations
from pyomo.environ import (
    AbstractModel,
    SolverFactory,
    value,
    Set,
    Param,
    Var,
    Constraint,
    Objective,
    Reals,
    Any,
    Integers,
    Binary,
    NonNegativeIntegers,
)
from .utils import add_equations, suppress_output, ext_visualise_output
from typing import Optional
from h2_plan.data import DefaultParams

# Planning import moved to function level to avoid circular imports
import logging
import matplotlib

#matplotlib.use("TkAgg")


def find_equivalent_set(model, values):
    values_set = set(values)
    for component in model.component_objects(Set, active=True):
        if set(component.data()) == values_set:
            return component
    return None


class MPCController:
    """ """

    def __init__(self):
        self.model1 = AbstractModel()
        self.model2 = AbstractModel()
        self._update_keys = None
        self._fig = None
        self._axs = None
        self._run_count = 0
        # Optimization: Track instances and solver state
        self.instance1 = None
        self.instance2 = None
        self.solver = None
        self._instances_created = False
        self._solver_configured = False
        # Solution caching for warm starts
        self._solution_cache = {}
        self._cache_enabled = True
        pass

    def render(self):
        if self._fig is None:
            import matplotlib.pyplot as plt

            self._fig, self._axs = plt.subplots(4, 1, figsize=(14.0, 7.0), sharex=True)
        return self._fig

    def build(self, data: dict):
        """
        This function builds the MPC problem
        """
        for key, set_ in data["sets"].items():
            setattr(self.model1, key, Set(initialize=list(set_)))
            getattr(self.model1, key).construct()
            setattr(self.model2, key, Set(initialize=list(set_)))
            getattr(self.model2, key).construct()

        for key, param in data["params"].items():
            if isinstance(param, dict):
                index_set_values = list(param.keys())

                for model in [self.model1, self.model2]:
                    existing_set = find_equivalent_set(model, index_set_values)

                    if existing_set is not None:
                        index_set = existing_set
                    else:
                        set_name = f"{key}_index"
                        index_set = Set(initialize=index_set_values, ordered=True)
                        setattr(model, set_name, index_set)

                    if not hasattr(model, key):
                        setattr(
                            model, key, Param(index_set, initialize=param, within=Any)
                        )
                        getattr(model, key).construct()
            else:
                setattr(self.model1, key, Param(initialize=param, within=Any))
                getattr(self.model1, key).construct()
                setattr(self.model2, key, Param(initialize=param, within=Any))
                getattr(self.model2, key).construct()

        for key, var in data["vars"].items():
            setattr(self.model1, key, Var(*var["time_duration"], within=var["domain"]))
            getattr(self.model1, key).construct()
            setattr(self.model2, key, Var(*var["time_duration"], within=var["domain"]))
            getattr(self.model2, key).construct()

        for key, constraint in data["constraints"].items():
            if key in data["forms"]["primary"]:
                setattr(
                    self.model1,
                    key,
                    Constraint(*constraint["time_duration"], rule=constraint["rule"]),
                )
                # getattr(self.model1, key).construct()
            if key in data["forms"]["secondary"]:
                setattr(
                    self.model2,
                    key,
                    Constraint(*constraint["time_duration"], rule=constraint["rule"]),
                )
                # getattr(self.model2, key).construct()

        for key, equation in data["equations"].items():
            if key in data["forms"]["primary"]:
                setattr(
                    self.model1,
                    key,
                    Constraint(*equation["time_duration"], rule=equation["rule"]),
                )
                # getattr(self.model1, key).construct()
            if key in data["forms"]["secondary"]:
                setattr(
                    self.model2,
                    key,
                    Constraint(*equation["time_duration"], rule=equation["rule"]),
                )
                # getattr(self.model2, key).construct()

        for key, objective in data["objectives"].items():
            if key in data["forms"]["primary"]:
                setattr(
                    self.model1,
                    key,
                    Objective(expr=objective["rule"], sense=objective["sense"]),
                )
                # getattr(self.model1, key).construct()
            if key in data["forms"]["secondary"]:
                setattr(
                    self.model2,
                    key,
                    Objective(expr=objective["rule"], sense=objective["sense"]),
                )
                # getattr(self.model2, key).construct()
        # Setting the fixed variable boolean to False as this will be the first solve
        setattr(self.model1, "fixed", Param(initialize=False, mutable=True))
        getattr(self.model1, "fixed").construct()
        setattr(self.model2, "fixed", Param(initialize=False, mutable=True))
        getattr(self.model2, "fixed").construct()

        pass

    def solve(self, supress=False, solver='gurobi',):
        """
        This function solves the MPC problem with optimized instance reuse
        """
        # Create instances only once
        #if not self._instances_created:
        self.instance1 = self.model1.create_instance()
        self.instance2 = self.model2.create_instance()
            #self._instances_created = True

        # Configure solver only once with performance optimizations

        if not self._solver_configured:
            self.solver = SolverFactory(solver)
            # Existing options
            self.solver.options["mipgap"] = 0.05
            self.solver.options["FeasibilityTol"] = 1e-6
            self.solver.options["OptimalityTol"] = 1e-9
            # Performance optimizations
            self.solver.options["Presolve"] = 2  # Aggressive presolve
            self.solver.options["Cuts"] = -1  # Conservative cuts
            self.solver.options["Heuristics"] = 0.1  # Reduced heuristics time
            self.solver.options["TimeLimit"] = 300  # 5 minute limit
            self.solver.options["Threads"] = 0  # Use all available cores
            self.solver.options["Method"] = 2  # Barrier method for LP relaxations
            self._solver_configured = True

        # Apply cached solution as warm start if available
        #if self._cache_enabled and self._solution_cache:
        #    self._apply_warm_start(self.instance1)

        with suppress_output(supress):
            self.results = self.solver.solve(
                self.instance1, tee=supress, warmstart=True
            )

        self.lexicographic = 1

        if self.results.solver.termination_condition != "optimal":
            print("[INFO] Infeasible problem, solving lexicographically")
            # Apply warm start to secondary model too
            if self._cache_enabled and self._solution_cache:
                self._apply_warm_start(self.instance2)
            self.results = self.solver.solve(
                self.instance2, tee=supress, warmstart=True
            )
            self.lexicographic = 2

        # Cache the solution for next iteration
        if (
            self._cache_enabled
            and self.results.solver.termination_condition == "optimal"
        ):
            self._cache_solution()

        return self.output()

    def update(self, stochastic_values, start_values: Optional[dict] = None):
        """
        This function updates the MPC problem
        """

        if start_values is None:
            self.stochastic_update(data=stochastic_values)
            return None

        for var in self.model1.component_objects(Var, active=True):
            if var.name == "cumulative_profit":
                continue
            for index in var:
                key = (var.name, index)
                if key in start_values:
                    # Clean the value based on variable domain before setting
                    cleaned_value = self._clean_variable_value(
                        var[index], start_values[key]
                    )
                    var[index].fix(cleaned_value)

        for var in self.model2.component_objects(Var, active=True):
            if var.name == "cumulative_profit":
                continue
            for index in var:
                key = (var.name, index)
                if key in start_values:
                    # Clean the value based on variable domain before setting
                    cleaned_value = self._clean_variable_value(
                        var[index], start_values[key]
                    )
                    var[index].fix(cleaned_value)
        
        self.stochastic_update(data=stochastic_values)

        return None

    def output(self, time_step: int = 24):
        solve = self.instance1 if self.lexicographic == 1 else self.instance2
    
        end_states = {}
        stochastic_output = {}
        stored_val = 0.0  # make it explicitly float
    
        # Extract ships sent / ordered
        stochastic_output["ordered_ship"] = {
            s: sum(
                value(getattr(solve, "n_ship_ordered")[s, t])
                for t in range(0, time_step, 24)
            )
            for s in getattr(solve, "ships", [])
        }
        stochastic_output["sent_ship"] = {
            s: sum(
                value(getattr(solve, "n_ship_sent")[s, t])
                for t in range(0, time_step, 24)
            )
            for s in getattr(solve, "ships", [])
        }
    
        # Total sent volume (scalar)
        sent_vol_by_ship = {
            s: sum(
                value(getattr(solve, "n_ship_sent")[s, t])
                for t in range(0, time_step, 24)
            ) * value(getattr(solve, "ship_capacity")[s])
            for s in getattr(solve, "ships", [])
        }
        
    
        # Use last time index instead of hardcoding 23
        last_t = time_step
    
        # Get numeric values for all terms
        hs = value(getattr(solve, "hydrogen_storage")[last_t])
        vs = value(getattr(solve, "vector_storage")[last_t])
        cc = value(getattr(solve, "cumulative_charge")[last_t])
        cv = value(getattr(solve, "calorific_value"))
        hs0 = value(getattr(solve, "hydrogen_storage")[0])
        vs0 = value(getattr(solve, "vector_storage")[0])
        cc0 = value(getattr(solve, "cumulative_charge")[0])
    
        # Build excess storage into long term rewards mechanis
        stored_val += (hs - hs0) / 120.0  # GJ / GJ/t(h2) = t(H2) 
        stored_val += (vs - vs0) * 1000.0 * cv / 120.0  # kt * t/kt * GJ/t / (GJ/t) = t(H2)
        stored_val += (cc - cc0) * cv / 120.0 # t * GJ/t) / (GJ/t) = t(H2)
    
        # Adjusting sent vol to tonnes of h2 eq from t(vect)
        sent_vol = sum(sent_vol_by_ship.values())  * cv /120 # t(vect) * GJ/t(vect) * t(h2) / GJ = t(H2)
        
        for var in solve.component_objects(Var):
            for index in var:
                if isinstance(index, tuple):
                    if index[-1] == time_step:  # assuming time is the last index
                        new_index = index[:-1] + (0,)  # change t=24 to t=0
                        var_value = value(var[index])
                        # Clean up numerical noise for different variable domains
                        var_value = self._clean_variable_value(var[index], var_value)
                        end_states[(var.name, new_index)] = var_value
                elif index == time_step:
                    var_value = value(var[index])
                    # Clean up numerical noise for different variable domains
                    var_value = self._clean_variable_value(var[index], var_value)
                    end_states[(var.name, 0)] = var_value

        getattr(self.model1, "fixed").set_value(True)
        getattr(self.model2, "fixed").set_value(True)

        return end_states, stochastic_output, stored_val, sent_vol

    def _clean_variable_value(self, var, value):
        """
        Clean variable values to handle numerical noise based on variable domain

        Args:
            var: Pyomo variable object
            value: Current variable value

        Returns:
            Cleaned value appropriate for the variable domain
        """
        from pyomo.environ import NonNegativeReals, NonNegativeIntegers, Integers

        # Handle None values
        if value is None:
            return 0.0

        # For integer domains, round to nearest integer
        if var.domain in [NonNegativeIntegers, Integers]:
            # Handle very small values that should be zero
            if abs(value) < 1e-12:
                cleaned_value = 0
            else:
                cleaned_value = int(round(value))

            # Ensure non-negative for NonNegativeIntegers
            if var.domain == NonNegativeIntegers and cleaned_value < 0:
                cleaned_value = 0
            return cleaned_value

        # For non-negative real domains, ensure non-negative
        elif var.domain == NonNegativeReals:
            # Clean up small negative values due to numerical noise
            if abs(value) < 1e-12:  # More aggressive tolerance for numerical noise
                return 0.0
            elif value < 0:
                # For larger negative values, clamp to zero but warn
                if abs(value) > 1e-3:
                    logging.warning(
                        f"Variable {var.name} has negative value {value}, clamping to 0"
                    )
                return 0.0
            else:
                return value

        # For general real domains, return as-is
        else:
            return value

    def stochastic_update(self, data: Optional[dict] = None):
        """
        Optimized function to update the MPC problem without parameter deletion
        """
        for key, param in data.items():
            if param["param"]["initialize"] is not None:
                # Try to update existing parameter values instead of deleting/recreating
                try:
                    if param["param"]["set"] is not None:
                        # Update parameter values for both models if they exist
                        param1 = getattr(self.model1, key, None)
                        param2 = getattr(self.model2, key, None)

                        if param1 is not None and hasattr(param1, "set_values"):
                            param1.set_values(param["param"]["initialize"])
                        else:
                            # Fallback to recreation if parameter doesn't exist
                            self._recreate_indexed_param(key, param)

                        if param2 is not None and hasattr(param2, "set_values"):
                            param2.set_values(param["param"]["initialize"])
                        else:
                            # Already recreated in _recreate_indexed_param
                            pass
                    else:
                        # Update scalar parameter
                        param1 = getattr(self.model1, key, None)
                        param2 = getattr(self.model2, key, None)

                        if param1 is not None and hasattr(param1, "set_value"):
                            param1.set_value(param["param"]["initialize"])
                        else:
                            self._recreate_scalar_param(key, param)

                        if param2 is not None and hasattr(param2, "set_value"):
                            param2.set_value(param["param"]["initialize"])
                        else:
                            # Already recreated in _recreate_scalar_param
                            pass

                except (AttributeError, KeyError):
                    # Fallback to original deletion/recreation approach
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
        pass

    def _recreate_indexed_param(self, key, param):
        """Helper method to recreate indexed parameters"""
        self.model1.del_component(key)
        self.model2.del_component(key)

        setattr(
            self.model1,
            key,
            Param(
                param["param"]["set"],
                initialize=param["param"]["initialize"],
                within=Reals,
                mutable=True,  # Make mutable for future updates
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
                mutable=True,  # Make mutable for future updates
            ),
        )
        getattr(self.model2, key).construct()

    def _recreate_scalar_param(self, key, param):
        """Helper method to recreate scalar parameters"""
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
        """
        Extracts latent states and dynamically updates plots across runs.
        """
        # Ensure figure and axes are created
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
        """Cache current solution for warm starting next iteration"""
        active_instance = self.instance1 if self.lexicographic == 1 else self.instance2

        self._solution_cache.clear()
        for var in active_instance.component_objects(Var, active=True):
            for index in var:
                if var[index].value is not None:
                    # Clean and store variable values for warm start
                    cleaned_value = self._clean_variable_value(
                        var[index], var[index].value
                    )
                    self._solution_cache[(var.name, index)] = cleaned_value

    def _apply_warm_start(self, instance):
        """Apply cached solution as warm start to given instance"""
        try:
            for var in instance.component_objects(Var, active=True):
                for index in var:
                    cache_key = (var.name, index)
                    if cache_key in self._solution_cache:
                        # The cached value should already be cleaned, but double-check
                        cached_value = self._solution_cache[cache_key]
                        cleaned_value = self._clean_variable_value(
                            var[index], cached_value
                        )
                        var[index].set_value(cleaned_value)
        except Exception as e:
            # If warm start fails, continue without it
            logging.warning(f"Warm start failed: {e}")

    def clear_cache(self):
        """Clear solution cache - useful for reset scenarios"""
        self._solution_cache.clear()

    def set_cache_enabled(self, enabled: bool):
        """Enable/disable solution caching"""
        self._cache_enabled = enabled
        if not enabled:
            self.clear_cache()
