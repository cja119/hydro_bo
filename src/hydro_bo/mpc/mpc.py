"""
MPC Controller Core Module
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any

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

from hydro_bo.utils.logging_config import get_logger
from hydro_bo.mpc.immutable import IMMUTABLE_MPC_PARAMS
from hydro_bo.mpc.utils import ext_visualise_output, suppress_output
from hydro_bo.mpc.contiguity import ContiguityHandler
from hydro_bo.mpc.relaxation import RelaxationTree

# Configure structured logger
logger = get_logger(__name__)

def ms(x):
    return int(x * 1000)

def minute_second(x):
    hours = int(x // 3600)
    minutes = int(x // 60)
    seconds = int(x % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

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
    def __init__(self, gurobi_seed=None):
        self.model = AbstractModel()
        self._update_keys = None
        self._fig = None
        self._axs = None
        self._run_count = 0
        self.instance = None
        self.solver = None
        self._solver_configured = False
        self._gurobi_seed = gurobi_seed
        self._instance_bound = False
        self._needs_instance_rebuild = False
        self._solution_cache = {}
        self._cache_enabled = True
        self._call_count = 0
        self._start_time = time.perf_counter()
        self.contiguity = ContiguityHandler()
        self.relaxation_tree = RelaxationTree()
        self._relaxation_stats = {"total_relaxations": 0, "relaxations_per_solve": [], "relaxations_by_type": {}}
        self._relaxation_baselines: Dict[str, Any] = {}
        self._relaxation_dirty = False
        self._last_handoff_cache = None
        self._current_uncertain_cache = None
        self._prev_uncertain_cache = None
        self._first_solve = True

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
        """This solves the MPCController model"""
        solve_start = time.perf_counter()

        solver_init_start = time.perf_counter()
        self._ensure_solver(solver)
        solver_init_time = time.perf_counter() - solver_init_start

        if solver_init_time > 0.001:
            logger.info("solver_initialization", duration_ms=ms(solver_init_time))

        if not supress:
            print(
                f"[DEBUG] Solve started, solver_init_time={solver_init_time*1000:.2f}ms",
                file=sys.stderr,
            )

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
                instance_creation_ms=ms(instance_time),
                solver_binding_ms=ms(bind_time),
                total_setup_ms=ms(instance_time + bind_time)
            )

            self._snapshot_relaxation_baselines()

        if self._relaxation_dirty:
            self._restore_relaxation_baselines()
        self.relaxation_tree.reset()
        relaxation_count = 0
        done = False

        while not done:
            solve_call_time, termination_condition = self._execute_solve_call(supress)

            if termination_condition == TerminationCondition.optimal:
                done = True
                continue

            if termination_condition == TerminationCondition.maxTimeLimit:
                if self._has_feasible_incumbent():
                    logger.info("accepting_time_limited_incumbent",
                                solve_call_ms=ms(solve_call_time))
                    done = True
                    continue
                logger.warning("time_limit_no_incumbent",
                               solve_call_ms=ms(solve_call_time))
                raise MPCSolveError(
                    termination_condition=termination_condition,
                    message=getattr(self.results.solver, "message", None),
                )

            relaxation_params = self.relaxation_tree.get_next_relaxation(str(termination_condition))

            if relaxation_params is None:
                print(f"[MPC] Relaxation tree exhausted after {relaxation_count} attempts, termination: {termination_condition}", file=sys.stderr, flush=True)

                self._dump_failed_initial_conditions(termination_condition)
                self._diagnose_infeasibility()

                raise MPCSolveError(
                    termination_condition=termination_condition,
                    message=getattr(self.results.solver, "message", None),
                )

            self._apply_relaxation_params(relaxation_params)
            self._relaxation_dirty = True
            relaxation_count += 1

        # Track which relaxation type actually succeeded (if any)
        if relaxation_count > 0 and self.relaxation_tree.last_applied is not None:
            successful_relaxation = self.relaxation_tree.last_applied["name"]
            by_type = self._relaxation_stats["relaxations_by_type"]
            by_type[successful_relaxation] = by_type.get(successful_relaxation, 0) + 1

        # Extract output
        result = self.output()

        total_time = time.perf_counter() - solve_start
        elapsed_seconds = time.perf_counter() - self._start_time

        self._call_count += 1

        # Track relaxation statistics
        self._relaxation_stats["relaxations_per_solve"].append(relaxation_count)
        self._relaxation_stats["total_relaxations"] += relaxation_count

        logger.info(
            "solve_complete",
            call_count=self._call_count,
            solve_call_ms=ms(solve_call_time),
            total_ms=ms(total_time),
            is_first_solve=not self._instance_bound,
            time_elapsed=minute_second(elapsed_seconds),
            relaxations_applied=relaxation_count,
            total_relaxations=self._relaxation_stats["total_relaxations"]
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
                rebuild_ms=ms(rebuild_time)
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
                refresh_ms=ms(refresh_time),
                has_start_values=False
            )
            return None

        # Order for gurobi_persistent:
        # 1 unfix all previously fixed variables
        # 2 update stochastic parameters
        # 3 fix initial conditions

        if self.instance is not None:
            num_unfixed = self.contiguity.unfix_all_variables(
                self.instance, self.solver, self._instance_bound
            )

        stochastic_start = time.perf_counter()
        self.stochastic_update(data=stochastic_values)
        stochastic_time = time.perf_counter() - stochastic_start

        # Cache the handoff values being applied; overwrite previous cache each window.
        self._last_handoff_cache = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "window_index": self._call_count,
            "start_values": {
                str(k): (v if isinstance(v, (int, float, bool, str, type(None))) else repr(v))
                for k, v in start_values.items()
            },
        }

        # Rotate uncertain-vector caches before overwriting with this window's values.
        self._prev_uncertain_cache = self._current_uncertain_cache
        self._current_uncertain_cache = self._extract_uncertain_vectors(stochastic_values)

        start_values_apply_start = time.perf_counter()
        # Only apply start values if instance exists (after first solve)
        if self.instance is not None:
            fixed_by_name = self.contiguity.apply_start_values(
                self.instance, self.solver, start_values,
                self._instance_bound, self._clean_variable_value
            )
        start_values_time = time.perf_counter() - start_values_apply_start

        refresh_start = time.perf_counter()
        self._refresh_persistent_solver_from_instance()
        refresh_time = time.perf_counter() - refresh_start

        total_time = time.perf_counter() - update_start

        logger.info(
            "parameter_update",
            rebuild_ms=ms(rebuild_time),
            refresh_ms=ms(refresh_time),
            total_update_ms=ms(total_time),
            has_start_values=True,
            start_vals=len(start_values)
        )

        return None

    def output(self, time_step: int = 24):
        """This grabs the conditional variables for the uncertainty."""

        solve = self.instance

        # Extract end states and outputs
        end_states = {}
        stochastic_output = {}
        stored_val = 0.0

        # Extract ships sent / ordered.
        # t=0 is only counted on the first window (no prior window to inherit from).
        t_start = 0 if self._first_solve else 24
        self._first_solve = False
        ordered_ship, sent_ship = self._ship_orders(solve, time_step, t_start)
        sent_vol_by_ship = self._sent_volumes(solve, time_step, t_start)

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
        from pyomo.environ import Binary, Integers, NonNegativeIntegers, NonNegativeReals

        if value is None:
            return 0.0

        if var.domain == Binary:
            return 1 if value >= 0.5 else 0

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

    def get_relaxation_stats(self) -> Dict[str, Any]:
        """Get relaxation statistics for reporting.

        Returns:
            Dictionary with relaxation statistics including:
            - total_relaxations: Total relaxations across all solves
            - relaxations_per_solve: List of relaxation counts per solve
            - avg_relaxations: Average relaxations per solve
            - max_relaxations: Maximum relaxations in a single solve
        """
        stats = dict(self._relaxation_stats)
        if stats["relaxations_per_solve"]:
            stats["avg_relaxations"] = sum(stats["relaxations_per_solve"]) / len(stats["relaxations_per_solve"])
            stats["max_relaxations"] = max(stats["relaxations_per_solve"])
        else:
            stats["avg_relaxations"] = 0.0
            stats["max_relaxations"] = 0

        return stats

    def log_relaxation_summary(self):
        """Log per-type relaxation usage as % of solve windows."""
        total_windows = self._call_count
        if total_windows == 0:
            return

        by_type = self._relaxation_stats["relaxations_by_type"]
        if not by_type:
            logger.info("relaxation_summary", total_windows=total_windows,
                        message="no relaxations used")
            return

        summary = {}
        for rtype, count in by_type.items():
            pct = (count / total_windows) * 100
            summary[rtype] = {"count": count, "pct": round(pct, 1)}

        logger.info("relaxation_summary",
                    total_windows=total_windows,
                    by_type=summary)

        # Human-readable stderr output
        print(f"[MPC] Relaxation summary ({total_windows} windows):", file=sys.stderr, flush=True)
        for rtype, info in summary.items():
            print(f"  {rtype}: {info['count']}/{total_windows} windows ({info['pct']}%)",
                  file=sys.stderr, flush=True)

    def stochastic_update(self, data: Optional[dict] = None):
        """Update stochastic parameters using the ContiguityHandler handler."""
        params_updated = self.contiguity.apply_parameter_updates(
            self.instance, self.solver, data, self._instance_bound
        )

        # Sync update keys from relaxation handler
        self._update_keys = self.contiguity.get_update_keys()

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


    def apply_param_updates(self, updates: Dict[str, Any]) -> None:
        """Set mutable parameter values on the built instance in place and
        refresh the persistent solver — no Pyomo instance rebuild."""
        if self.instance is None or not self._instance_bound:
            raise RuntimeError("apply_param_updates requires a built, bound instance")
        for name, val in updates.items():
            if name in IMMUTABLE_MPC_PARAMS:
                raise ValueError(f"{name!r} is structural; rebuild required")
            param_obj = getattr(self.instance, name, None)
            if param_obj is None:
                raise KeyError(f"unknown MPC parameter: {name!r}")
            if isinstance(val, dict):
                for idx, v in val.items():
                    param_obj[idx].set_value(v)
            else:
                param_obj.set_value(val)
        self._refresh_persistent_solver_from_instance()

    def _refresh_persistent_solver_from_instance(self):
        """Rebuild persistent solver view from current instance state."""
        if self.instance is None or not self._instance_bound:
            return
        if not hasattr(self.solver, "set_instance"):
            return

        self.solver.set_instance(self.instance)

    def _execute_solve_call(self, supress: bool = True):
        """Execute the solver call with suppressed output and warning handling."""
        solve_call_start = time.perf_counter()
        with suppress_output(supress):
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Setting Var.*not in domain")
                warnings.filterwarnings("ignore", message="Loading a SolverResults object with a warning status")
                warnings.filterwarnings("ignore", category=UserWarning, module="pyomo.core")
                self.results = self.solver.solve(tee=not supress, warmstart=False)
        solve_call_time = time.perf_counter() - solve_call_start

        return solve_call_time, self.results.solver.termination_condition

    def _apply_relaxation_params(self, relaxation_params: Dict[str, Any]):
        """Apply relaxation parameters to the model instance."""
        if self.instance is None:
            logger.warning("cannot_apply_relaxation", reason="instance_not_created")
            return

        for param_name, new_value in relaxation_params.items():
            try:
                param_obj = getattr(self.instance, param_name, None)

                if param_obj is None:
                    logger.warning(
                        "relaxation_param_not_found",
                        param=param_name,
                        value=new_value
                    )
                    continue

                # Update parameter value
                if hasattr(param_obj, "set_value"):
                    param_obj.set_value(new_value)
                else:
                    param_obj.value = new_value

                logger.info(
                    "relaxation_param_applied",
                    param=param_name,
                    value=new_value
                )

            except Exception as e:
                logger.error(
                    "relaxation_param_failed",
                    param=param_name,
                    value=new_value,
                    error=str(e)
                )

        self._refresh_persistent_solver_from_instance()
        logger.info("persistent_solver_refreshed_after_relaxation")

    def _has_feasible_incumbent(self) -> bool:
        """Check whether Gurobi holds a feasible incumbent solution.

        Used after a maxTimeLimit termination to decide whether to accept
        the time-limited result rather than escalate to constraint relaxation.
        """
        gurobi_model = getattr(self.solver, "_solver_model", None)
        if gurobi_model is None:
            return False
        try:
            return int(gurobi_model.SolCount) > 0
        except Exception as e:
            logger.warning("incumbent_check_failed", error=str(e))
            return False

    def _snapshot_relaxation_baselines(self):
        """Snapshot baseline values for every parameter the tree may mutate.

        Called once, right after the instance is first built. The stored values
        are the canonical defaults that `_restore_relaxation_baselines` reverts
        to at the start of each subsequent solve.
        """
        if self.instance is None:
            return

        self._relaxation_baselines = {}
        for name in self.relaxation_tree.get_relaxable_param_names():
            param_obj = getattr(self.instance, name, None)
            if param_obj is None:
                continue
            try:
                self._relaxation_baselines[name] = value(param_obj)
            except Exception as e:
                logger.warning("relaxation_baseline_snapshot_failed", param=name, error=str(e))

        logger.info("relaxation_baselines_snapshot", params=list(self._relaxation_baselines))

    def _restore_relaxation_baselines(self):
        """Revert any parameters a prior relaxed solve left mutated."""
        if not self._relaxation_baselines:
            self._relaxation_dirty = False
            return

        for name, baseline in self._relaxation_baselines.items():
            param_obj = getattr(self.instance, name, None)
            if param_obj is None:
                continue
            if hasattr(param_obj, "set_value"):
                param_obj.set_value(baseline)
            else:
                param_obj.value = baseline

        self._refresh_persistent_solver_from_instance()
        self._relaxation_dirty = False
        logger.info("relaxation_baselines_restored", params=list(self._relaxation_baselines))

    def _build_sets(self, sets_def):
        for key, set_ in sets_def.items():
            setattr(self.model, key, Set(initialize=list(set_)))
            getattr(self.model, key).construct()

    def _build_params(self, params_def):
        for key, param in params_def.items():
            is_mutable = key not in IMMUTABLE_MPC_PARAMS

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

        # h2_price: scalar hydrogen price ($/kg), updated each day via PriceDynamics
        if not hasattr(self.model, "h2_price"):
            setattr(
                self.model,
                "h2_price",
                Param(initialize=5.0, within=Any, mutable=True)
            )
            getattr(self.model, "h2_price").construct()

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
        opts["Threads"] = 1
        opts["Method"] = 1
        if self._gurobi_seed is not None:
            from hydro_bo.utils.seeding import gurobi_seed as _clamp_gurobi_seed
            opts["Seed"] = _clamp_gurobi_seed(self._gurobi_seed)
        self._solver_configured = True

    def _ship_orders(self, solve, time_step, t_start):
        ordered_ship = {
            s: sum(
                value(getattr(solve, "n_ship_ordered")[s, t])
                for t in range(t_start, time_step + 1, 24)
            )
            for s in getattr(solve, "ships", [])
        }
        sent_ship = {
            s: sum(
                value(getattr(solve, "n_ship_sent")[s, t])
                for t in range(t_start, time_step + 1, 24)
            )
            for s in getattr(solve, "ships", [])
        }
        return ordered_ship, sent_ship

    def _sent_volumes(self, solve, time_step, t_start):
        return {
            s: sum(
                value(getattr(solve, "n_ship_sent")[s, t])
                for t in range(t_start, time_step + 1, 24)
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

    def _extract_uncertain_vectors(self, stochastic_values):
        """Serialise a stochastic-values dict for JSON storage.

        Handles nested dicts (e.g. {(ship, t): val}) and plain scalars.
        Returns a dict with timestamp/window_index metadata plus the vectors.
        """
        def _safe(v):
            return v if isinstance(v, (int, float, bool, str, type(None))) else repr(v)

        serialised = {}
        for k, v in stochastic_values.items():
            if isinstance(v, dict):
                serialised[str(k)] = {str(ki): _safe(vi) for ki, vi in v.items()}
            else:
                serialised[str(k)] = _safe(v)

        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "window_index": self._call_count,
            "uncertain_vectors": serialised,
        }

    def _dump_failed_initial_conditions(self, termination_condition):
        """Write the cached handoff initial conditions and uncertain vectors to JSON for debugging."""
        out_dir = os.path.join("debug_mpc")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "failed_initial_conditions.json")

        payload = {
            "termination_condition": str(termination_condition),
            "failed_at_timestamp": datetime.utcnow().isoformat() + "Z",
            "failed_at_window_index": self._call_count,
            "handoff_from_previous_window": self._last_handoff_cache,
            "current_window_uncertain_vectors": self._current_uncertain_cache,
            "previous_window_uncertain_vectors": self._prev_uncertain_cache,
        }

        try:
            with open(out_path, "w") as fh:
                json.dump(payload, fh, indent=2)
            print(f"[MPC] Failed initial conditions written to {out_path}", file=sys.stderr, flush=True)
        except Exception as exc:
            print(f"[MPC] Could not write failed_initial_conditions.json: {exc}", file=sys.stderr, flush=True)

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

            self._estimate_feasibility_distance(gurobi_model)

        except Exception as e:
            logger.error("iis_computation_failed", error=str(e))
            print(f"[ERROR] Failed to compute IIS: {e}", file=sys.stderr)

    def _estimate_feasibility_distance(self, gurobi_model):
        """Estimate how close the model is to feasibility via feasRelax"""
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

