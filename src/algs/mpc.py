"""
MPC Controller Core Module
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Optional

from pyomo.environ import (
    AbstractModel,
    Any,
    Constraint,
    Objective,
    Param,
    Set,
    SolverFactory,
    Var,
    value,
)
from pyomo.opt import TerminationCondition

from algs.logging_config import get_logger
from algs.utils import ext_visualise_output, suppress_output

# Configure structured logger
logger = get_logger(__name__)

def ms(x):
    return round(x * 1000, 1)

class MPCSolveError(RuntimeError):
    def __init__(self, termination_condition, message=None):
        self.termination_condition = termination_condition
        self.message = message
        msg_str = f" - {str(message)}" if message is not None else ""
        super().__init__(
            f"MPC solve failed: {termination_condition}{msg_str}"
        )


def find_equivalent_set(model, values):
    values_set = set(values)
    for component in model.component_objects(Set, active=True):
        if set(component.data()) == values_set:
            return component
    return None


class MPCController:
    def __init__(self):
        self.model = AbstractModel()
        self._update_keys = None
        self._fig = None
        self._axs = None
        self._run_count = 0
        self.instance = None
        self.solver = None
        self._solver_configured = False
        self._instance_bound = False
        self._needs_instance_rebuild = False
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
        self._build_stochastic_params(data["sets"])  # Add placeholder stochastic params
        self._build_vars(data["vars"])
        self._build_constraints(data["constraints"], data["forms"])
        self._build_equations(data["equations"], data["forms"])
        self._build_objectives(data["objectives"], data["forms"])
        self._init_fixed_flags()

    def solve(self, supress: bool = True, solver="gurobi"):
        """
        This solves the MPCController model.

        :param self: MPCController instance
        :param supress: Whether to suppress solver/stdout output
        :param solver: Solver name (ignored, always uses gurobi_persistent)
        """
        solve_start = time.perf_counter()

        # Initialize solver if needed
        solver_init_start = time.perf_counter()
        self._ensure_solver(solver)
        solver_init_time = time.perf_counter() - solver_init_start

        if solver_init_time > 0.001:
            logger.info("solver_initialization", duration_ms=solver_init_time * 1000)

        if not supress:
            print(
                f"[DEBUG] Solve started, solver_init_time={solver_init_time*1000:.2f}ms",
                file=sys.stderr,
            )

        # On first solve, create instance and bind to persistent solver
        if not self._instance_bound:
            instance_start = time.perf_counter()
            self.instance = self.model.create_instance()
            instance_time = time.perf_counter() - instance_start

            bind_start = time.perf_counter()
            self.solver.set_instance(self.instance)
            bind_time = time.perf_counter() - bind_start

            self._instance_bound = True

            logger.info(
                "first_solve_setup",
                instance_creation_ms=instance_time * 1000,
                solver_binding_ms=bind_time * 1000,
                total_setup_ms=(instance_time + bind_time) * 1000
            )

        # Solve model (persistent solver doesn't need instance parameter)
        solve_call_start = time.perf_counter()
        with suppress_output(supress):
            self.results = self.solver.solve(tee=not supress, warmstart=False)
        solve_call_time = time.perf_counter() - solve_call_start

        # Check if optimal
        if self.results.solver.termination_condition != TerminationCondition.optimal:
            logger.warning(
                "solve_failed",
                termination_condition=str(self.results.solver.termination_condition),
                solve_time_ms=solve_call_time * 1000
            )

            # If infeasible, write LP file and compute IIS for debugging
            tc = self.results.solver.termination_condition
            if tc == TerminationCondition.infeasible or tc == TerminationCondition.infeasibleOrUnbounded:
                self._diagnose_infeasibility()

            raise MPCSolveError(
                termination_condition=self.results.solver.termination_condition,
                message=getattr(self.results.solver, "message", None),
            )
        # Extract output
        output_start = time.perf_counter()
        result = self.output()
        output_time = time.perf_counter() - output_start

        total_time = time.perf_counter() - solve_start

        logger.info(
            "solve_complete",
            solve_call_ms=round(solve_call_time * 1000, 1),
            output_ms=round(output_time * 1000, 1),
            total_ms=round(total_time * 1000, 1),
            is_first_solve=not self._instance_bound or solve_call_time > 1.0
        )

        return result

    def update(self, stochastic_values, start_values: Optional[dict] = None):
        """
        Update the MPCController with new stochastic and starting values.

        :param self: MPCController instance
        :param stochastic_values: Stochastic values for update
        :param start_values: Starting values for update
        :type start_values: Optional[dict]
        """
        update_start = time.perf_counter()

        # When `fixed` first flips to True, several equation rules switch to
        # Constraint.Skip at t=0. With a persistent solver, those structural
        # changes require rebuilding the concrete instance once.
        rebuild_time = 0.0
        if self._needs_instance_rebuild and self.instance is not None:
            rebuild_start = time.perf_counter()
            self.instance = self.model.create_instance()
            self.solver.set_instance(self.instance)
            self._instance_bound = True
            self._needs_instance_rebuild = False
            rebuild_time = time.perf_counter() - rebuild_start
            logger.info(
                "instance_rebuilt_for_fixed_mode",
                rebuild_ms=rebuild_time * 1000,
            )

        # If no start values, just do stochastic update
        if start_values is None:
            stochastic_start = time.perf_counter()
            self.stochastic_update(data=stochastic_values)
            stochastic_time = time.perf_counter() - stochastic_start

            refresh_start = time.perf_counter()
            self._refresh_persistent_solver_from_instance()
            refresh_time = time.perf_counter() - refresh_start

            logger.info(
                "parameter_update",
                stochastic_update_ms=stochastic_time * 1000,
                refresh_ms=refresh_time * 1000,
                rebuild_ms=rebuild_time * 1000,
                has_start_values=False
            )
            return None

        # CRITICAL ORDER for gurobi_persistent compatibility:
        # 1. Unfix all previously fixed variables
        # 2. Update stochastic parameters (changes constraints)
        # 3. Fix initial conditions with new parameter values

        unfix_start = time.perf_counter()
        if self.instance is not None:
            self._unfix_all_variables(self.instance)
        unfix_time = time.perf_counter() - unfix_start

        stochastic_start = time.perf_counter()
        self.stochastic_update(data=stochastic_values)
        stochastic_time = time.perf_counter() - stochastic_start

        start_values_apply_start = time.perf_counter()
        # Only apply start values if instance exists (after first solve)
        if self.instance is not None:
            self._apply_start_values(self.instance, start_values)
        start_values_time = time.perf_counter() - start_values_apply_start

        refresh_start = time.perf_counter()
        self._refresh_persistent_solver_from_instance()
        refresh_time = time.perf_counter() - refresh_start

        total_time = time.perf_counter() - update_start

        logger.info(
            "parameter_update",
            rebuild_ms=ms(rebuild_time),
            unfix_ms=ms(unfix_time),
            stochastic_update_ms=ms(stochastic_time),
            start_values_ms=ms(start_values_time),
            refresh_ms=ms(refresh_time),
            total_update_ms=ms(total_time),
            has_start_values=True,
            num_start_values=len(start_values)
        )

        return None

    def output(self, time_step: int = 24):
        """
        This grabs the conditional variables for the uncertainty.

        :param self: MPCController instance
        :param time_step: Time step for output
        :type time_step: int
        """

        solve = self.instance

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
        # CRITICAL: also mirror to abstract model so rebuilt instances inherit it.
        if self.instance is not None:
            fixed_param = getattr(self.instance, "fixed")
            was_fixed = bool(value(fixed_param))

            getattr(self.instance, "fixed").set_value(True)
            getattr(self.model, "fixed").set_value(True)

            # Equation rules branch on m.fixed at construction time.
            # Trigger a one-time rebuild when transitioning False -> True.
            if not was_fixed:
                self._needs_instance_rebuild = True
                logger.info("fixed_parameter_changed", action="constraints_will_regenerate_on_next_solve")
        else:
            getattr(self.model, "fixed").set_value(True)

        return end_states, stochastic_output, stored_val, sent_vol

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

    def clear_cache(self):
        self._solution_cache.clear()

    def set_cache_enabled(self, enabled: bool):
        self._cache_enabled = enabled
        if not enabled:
            self.clear_cache()

    def stochastic_update(self, data: Optional[dict] = None):
        # If instance doesn't exist yet, we can't update parameters
        # The first solve will create the instance with current parameter values
        if self.instance is None:
            logger.info("stochastic_update_skipped", reason="instance_not_created")
            return

        # Update parameters on the instance
        target = self.instance
        params_updated = {}

        for key, param in data.items():
            if param["param"]["initialize"] is not None:
                try:
                    param_obj = getattr(target, key, None)

                    if param_obj is None:
                        raise RuntimeError(
                            f"Parameter '{key}' not found on instance. "
                            f"This indicates a structural change which is incompatible with gurobi_persistent."
                        )

                    if param["param"]["set"] is not None:
                        # For indexed parameters, update each index individually
                        init_data = param["param"]["initialize"]
                        param_set = param["param"]["set"]

                        if isinstance(init_data, dict):
                            # Dict format: keys are indices
                            for idx, val in init_data.items():
                                if idx in param_obj:
                                    param_obj[idx] = val
                        elif isinstance(init_data, (list, tuple)):
                            # List/tuple format: indices come from the set
                            if isinstance(param_set, (list, tuple)):
                                for idx, val in zip(param_set, init_data):
                                    if idx in param_obj:
                                        param_obj[idx] = val
                            else:
                                raise RuntimeError(
                                    f"Parameter '{key}' has list/tuple initialize but set is not a list/tuple. "
                                    f"This indicates a structural change which is incompatible with gurobi_persistent."
                                )
                        else:
                            raise RuntimeError(
                                f"Parameter '{key}' has a set but initialize is not a dict/list/tuple. "
                                f"This indicates a structural change which is incompatible with gurobi_persistent."
                            )
                    else:
                        # For scalar parameters, use set_value
                        if hasattr(param_obj, "set_value"):
                            param_obj.set_value(param["param"]["initialize"])
                        else:
                            # Scalar parameter, direct assignment
                            param_obj.value = param["param"]["initialize"]

                    # Track parameter updates
                    signature = self._parameter_signature(init_data)
                    params_updated[key] = {
                        "type": "indexed" if param["param"]["set"] is not None else "scalar",
                        "num_values": len(init_data) if isinstance(init_data, (dict, list, tuple)) else 1,
                        "signature": signature,
                    }

                except (AttributeError, KeyError) as e:
                    raise RuntimeError(
                        f"Failed to update parameter '{key}': {e}. "
                        f"This indicates a structural change which is incompatible with gurobi_persistent. "
                        f"All parameter updates must be in-place (no structural changes)."
                    ) from e

                # These keys will be used to grab the output
                if param["loc"] == "endogenous":
                    if self._update_keys is None:
                        self._update_keys = {key: param["name"]}
                    elif key not in self._update_keys:
                        self._update_keys[key] = param["name"]

        if self._instance_bound and hasattr(self.solver, '_solver_model'):
            try:
                gurobi_model = self.solver._solver_model
                gurobi_model.update()
                logger.info("gurobi_model_updated_after_params")
            except Exception as e:
                logger.warning("failed_to_update_gurobi_model", error=str(e))

    def visualise_output(self, time_step: int = 24):
        if self._fig is None:
            import matplotlib.pyplot as plt

            self._fig, self._axs = plt.subplots(4, 1, figsize=(14.0, 7.0), sharex=True)

        solve = self.instance

        self._joined_data = ext_visualise_output(
            solve=solve,
            axs=self._axs,
            run_count=self._run_count,
            time_step=time_step,
            joined_data=self._joined_data if hasattr(self, "_joined_data") else None,
            fig=self._fig,
        )

        self._run_count += 1

    def _apply_start_values(self, model, start_values):
        # Fix variables at t=0 to ensure continuity between MPC solves
        # These represent the end state from the previous solve's horizon
        fixed_vars = []
        fixed_by_name = {}

        for var in model.component_objects(Var, active=True):
            if var.name == "cumulative_profit":
                continue
            for index in var:
                key = (var.name, index)
                if key in start_values:
                    cleaned = self._clean_variable_value(var[index], start_values[key])
                    var[index].fix(cleaned)
                    fixed_vars.append(var[index])

                    # Track for logging
                    if var.name not in fixed_by_name:
                        fixed_by_name[var.name] = 0
                    fixed_by_name[var.name] += 1

        # Log what we fixed
        logger.info("fixed_initial_conditions",
                   total_vars_fixed=len(fixed_vars),
                   vars_by_name=fixed_by_name)

        # Notify persistent solver of fixed variables if instance is bound
        if self._instance_bound and hasattr(self.solver, 'update_var'):
            for var_obj in fixed_vars:
                self.solver.update_var(var_obj)

    def _unfix_all_variables(self, model):
        # Unfix all previously fixed variables before updating parameters
        # This prevents conflicts when parameters change between solves
        unfixed_vars = []
        for var in model.component_objects(Var, active=True):
            for index in var:
                if var[index].fixed:
                    var[index].unfix()
                    unfixed_vars.append(var[index])

        # Notify persistent solver of unfixed variables if instance is bound
        if self._instance_bound and hasattr(self.solver, 'update_var'):
            for var_obj in unfixed_vars:
                self.solver.update_var(var_obj)


    def _refresh_persistent_solver_from_instance(self):
        """Rebuild persistent solver view from current instance state.

        This ensures mutable parameter coefficient updates and fixed/unfixed status
        are consistently reflected in the backend model across MPC iterations.
        """
        if self.instance is None or not self._instance_bound:
            return
        if not hasattr(self.solver, "set_instance"):
            return

        self.solver.set_instance(self.instance)

    def _parameter_signature(self, init_data):
        """Return a compact numeric signature for update diagnostics."""
        if isinstance(init_data, dict):
            vals = [v for v in init_data.values() if isinstance(v, (int, float))]
            if not vals:
                return "non_numeric_dict"
            return {
                "min": float(min(vals)),
                "max": float(max(vals)),
                "sum": float(sum(vals)),
            }
        if isinstance(init_data, (list, tuple)):
            vals = [v for v in init_data if isinstance(v, (int, float))]
            if not vals:
                return "non_numeric_sequence"
            return {
                "min": float(min(vals)),
                "max": float(max(vals)),
                "sum": float(sum(vals)),
            }
        if isinstance(init_data, (int, float)):
            return float(init_data)
        return "non_numeric"

    def _build_sets(self, sets_def):
        for key, set_ in sets_def.items():
            setattr(self.model, key, Set(initialize=list(set_)))
            getattr(self.model, key).construct()

    def _build_params(self, params_def):
        # Parameters that must remain immutable (used in control flow)
        immutable_params = {'mean_ship_arrival_time', 'mean_ship_transit_time', 'std_ship_transit_time'}

        for key, param in params_def.items():
            # Determine if this parameter should be mutable
            is_mutable = key not in immutable_params

            if isinstance(param, dict):
                index_set_values = list(param.keys())
                index_set = find_equivalent_set(self.model, index_set_values)
                if index_set is None:
                    set_name = f"{key}_index"
                    index_set = Set(initialize=index_set_values, ordered=True)
                    setattr(self.model, set_name, index_set)
                if not hasattr(self.model, key):
                    setattr(
                        self.model, key, Param(index_set, initialize=param, within=Any, mutable=is_mutable)
                    )
                    getattr(self.model, key).construct()
            else:
                setattr(self.model, key, Param(initialize=param, within=Any, mutable=is_mutable))
                getattr(self.model, key).construct()

    def _build_stochastic_params(self, sets_def):
        """Build placeholder stochastic parameters that will be updated via stochastic_update."""
        # energy_wind: indexed by grid0 (time steps)
        if "grid0" in sets_def and not hasattr(self.model, "energy_wind"):
            default_values = {t: 0.0 for t in sets_def["grid0"]}
            setattr(
                self.model,
                "energy_wind",
                Param(getattr(self.model, "grid0"), initialize=default_values, within=Any, mutable=True)
            )
            getattr(self.model, "energy_wind").construct()

        # ship_arrived: indexed by ships
        if "ships" in sets_def and not hasattr(self.model, "ship_arrived"):
            default_values = {s: 0 for s in sets_def["ships"]}
            setattr(
                self.model,
                "ship_arrived",
                Param(getattr(self.model, "ships"), initialize=default_values, within=Any, mutable=True)
            )
            getattr(self.model, "ship_arrived").construct()

        # expected_ships: indexed by (ships, grid1)
        if "ships" in sets_def and "grid1" in sets_def and not hasattr(self.model, "expected_ships"):
            default_values = {(s, t): 0 for s in sets_def["ships"] for t in sets_def["grid1"]}
            setattr(
                self.model,
                "expected_ships",
                Param(
                    getattr(self.model, "ships"),
                    getattr(self.model, "grid1"),
                    initialize=default_values,
                    within=Any,
                    mutable=True
                )
            )
            getattr(self.model, "expected_ships").construct()

    def _build_vars(self, vars_def):
        for key, var in vars_def.items():
            setattr(self.model, key, Var(*var["time_duration"], within=var["domain"]))
            getattr(self.model, key).construct()

    def _build_constraints(self, constraints_def, forms):
        for key, constraint in constraints_def.items():
            if key in forms["primary"]:
                setattr(
                    self.model,
                    key,
                    Constraint(*constraint["time_duration"], rule=constraint["rule"]),
                )

    def _build_equations(self, equations_def, forms):
        for key, equation in equations_def.items():
            if key in forms["primary"]:
                setattr(
                    self.model,
                    key,
                    Constraint(*equation["time_duration"], rule=equation["rule"]),
                )

    def _build_objectives(self, objectives_def, forms):
        for key, objective in objectives_def.items():
            if key in forms["primary"]:
                setattr(
                    self.model,
                    key,
                    Objective(expr=objective["rule"], sense=objective["sense"]),
                )

    def _init_fixed_flags(self):
        setattr(self.model, "fixed", Param(initialize=False, mutable=True))
        getattr(self.model, "fixed").construct()

    def _ensure_solver(self, solver):
        if self._solver_configured:
            return

        # Use gurobi_persistent for optimal performance with repeated solves
        self.solver = SolverFactory('gurobi_persistent')

        # Configure Gurobi options
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

    def _diagnose_infeasibility(self):
        """
        Diagnose infeasibility by writing LP file and computing IIS.
        """
        from pathlib import Path

        # Create debug directory
        debug_dir = Path("./debug_mpc")
        debug_dir.mkdir(exist_ok=True)

        # Write LP file
        lp_file = debug_dir / "infeasible_model.lp"
        self.instance.write(str(lp_file), io_options={'symbolic_solver_labels': True})
        logger.error("infeasibility_detected", lp_file=str(lp_file))
        print(f"[ERROR] Infeasible model written to: {lp_file}", file=sys.stderr)

        # Compute IIS using Gurobi
        try:
            # Get the Gurobi model from the persistent solver
            gurobi_model = self.solver._solver_model

            # If we got "infeasibleOrUnbounded", disable dual reductions to clarify
            tc = self.results.solver.termination_condition
            if tc == TerminationCondition.infeasibleOrUnbounded:
                logger.info("clarifying_infeasible_or_unbounded", action="disabling_dual_reductions")
                print("[INFO] Got 'infeasible or unbounded' - disabling dual reductions to clarify...", file=sys.stderr)
                gurobi_model.setParam('DualReductions', 0)
                gurobi_model.optimize()
                print(f"[INFO] After disabling dual reductions: Status = {gurobi_model.Status}", file=sys.stderr)

            # Compute IIS
            gurobi_model.computeIIS()

            # Write IIS file
            iis_file = debug_dir / "infeasible_model.ilp"
            gurobi_model.write(str(iis_file))

            logger.error("iis_computed", iis_file=str(iis_file))
            print(f"[ERROR] IIS (Irreducible Inconsistent Subsystem) written to: {iis_file}", file=sys.stderr)

            # Collect IIS members once, then print + log them.
            iis_constraints = []
            iis_bounds = []

            for constr in gurobi_model.getConstrs():
                if constr.IISConstr:
                    iis_constraints.append(str(constr.ConstrName))

            for var in gurobi_model.getVars():
                if var.IISLB:
                    iis_bounds.append({
                        "var": str(var.VarName),
                        "bound_type": "LB",
                        "bound": float(var.LB),
                    })
                if var.IISUB:
                    iis_bounds.append({
                        "var": str(var.VarName),
                        "bound_type": "UB",
                        "bound": float(var.UB),
                    })

            # Keep terminal behavior for quick local visibility.
            print("\n[ERROR] Conflicting constraints in IIS:", file=sys.stderr)
            for name in iis_constraints:
                print(f"  - Constraint: {name}", file=sys.stderr)

            print("\n[ERROR] Conflicting bounds in IIS:", file=sys.stderr)
            for b in iis_bounds:
                label = "lower" if b["bound_type"] == "LB" else "upper"
                print(
                    f"  - Variable {b['var']} {label} bound: {b['bound']}",
                    file=sys.stderr,
                )

            # Write full IIS IDs into run.log in chunks to avoid giant single-line entries.
            logger.error(
                "iis_members_summary",
                constraint_count=len(iis_constraints),
                bound_count=len(iis_bounds),
                chunk_size=200,
            )

            chunk_size = 200
            for i in range(0, len(iis_constraints), chunk_size):
                logger.error(
                    "iis_constraints_chunk",
                    chunk_index=(i // chunk_size) + 1,
                    total_chunks=(len(iis_constraints) + chunk_size - 1) // chunk_size,
                    constraint_ids=iis_constraints[i : i + chunk_size],
                )

            for i in range(0, len(iis_bounds), chunk_size):
                logger.error(
                    "iis_bounds_chunk",
                    chunk_index=(i // chunk_size) + 1,
                    total_chunks=(len(iis_bounds) + chunk_size - 1) // chunk_size,
                    bounds=iis_bounds[i : i + chunk_size],
                )

            # Estimate distance-to-feasibility using L1 feasibility relaxation.
            self._estimate_feasibility_distance(gurobi_model)

        except Exception as e:
            logger.error("iis_computation_failed", error=str(e))
            print(f"[ERROR] Failed to compute IIS: {e}", file=sys.stderr)

    def _estimate_feasibility_distance(self, gurobi_model):
        """Estimate how close the model is to feasibility via feasRelax.

        Uses Gurobi's L1 relaxation objective (sum of weighted violations).
        Lower values indicate the model is closer to feasible.
        """
        try:
            relaxed = gurobi_model.copy()
            relaxed.setParam("OutputFlag", 0)

            # L1 relaxation objective over bounds and constraints.
            try:
                relaxed.feasRelaxS(0, False, True, True)
            except TypeError:
                # API signature compatibility fallback.
                relaxed.feasRelaxS(0, False, True, True, True)

            relaxed.optimize()
            status = int(relaxed.Status)

            if status == 2:  # OPTIMAL
                relax_obj = float(relaxed.ObjVal)
                logger.error(
                    "feasibility_distance_estimated",
                    l1_relaxation_distance=relax_obj,
                    interpretation=(
                        "0 means feasible; larger values indicate greater total violation "
                        "needed to recover feasibility"
                    ),
                )
                print(
                    f"[ERROR] Feasibility distance (L1 relaxation): {relax_obj:.6g}",
                    file=sys.stderr,
                )

                # Report the dominant relaxed rows/bounds to localize infeasibility.
                self._report_relaxation_violations(relaxed)
            else:
                logger.warning(
                    "feasibility_distance_unavailable",
                    status=status,
                )
        except Exception as e:
            logger.warning("feasibility_distance_failed", error=str(e))

    def _report_relaxation_violations(self, relaxed_model):
        """Print top constraint and bound violations from feasibility relaxation."""
        try:
            top_n = 10

            # FeasRelax introduces artificial variables whose values are exactly the
            # bound/row relaxations required for feasibility.
            # Typical names: ArtL_<var>, ArtU_<var>, ArtP_<constr>, ArtN_<constr>.
            constraint_slacks = {}
            bound_violations = []

            for var in relaxed_model.getVars():
                name = str(var.VarName)
                viol = float(var.X)
                if viol <= 1e-9 or not name.startswith("Art"):
                    continue

                if name.startswith("ArtL_"):
                    bound_violations.append((viol, name[5:], "LB", None, None))
                elif name.startswith("ArtU_"):
                    bound_violations.append((viol, name[5:], "UB", None, None))
                elif name.startswith("ArtP_"):
                    row = name[5:]
                    constraint_slacks[row] = constraint_slacks.get(row, 0.0) + viol
                elif name.startswith("ArtN_"):
                    row = name[5:]
                    constraint_slacks[row] = constraint_slacks.get(row, 0.0) + viol

            con_violations = sorted(
                ((viol, row) for row, viol in constraint_slacks.items()),
                reverse=True,
                key=lambda x: x[0],
            )
            bound_violations.sort(reverse=True, key=lambda x: x[0])

            logger.error(
                "feasibility_relaxation_top_violations",
                top_constraint_violations=con_violations[:top_n],
                top_bound_violations=bound_violations[:top_n],
            )

            if con_violations:
                print("[ERROR] Top relaxed constraints:", file=sys.stderr)
                for viol, name in con_violations[:top_n]:
                    print(f"  - {name}: violation={viol:.6g}", file=sys.stderr)

            if bound_violations:
                print("[ERROR] Top relaxed bounds:", file=sys.stderr)
                for viol, name, btype, bval, xval in bound_violations[:top_n]:
                    print(
                        f"  - {name} {btype}: violation={viol:.6g}",
                        file=sys.stderr,
                    )
        except Exception as e:
            logger.warning("feasibility_relaxation_report_failed", error=str(e))
