"""
Constraint Relaxation Module

This module handles parameter updates for the MPC controller, including
constraint relaxation strategies for handling infeasible solves.

"""

from typing import Optional, Dict, Any
from pyomo.environ import Var
from hydro_bo.algs.logging_config import get_logger

logger = get_logger(__name__)


class ContiguityHandler:
    """Handles parameter updates and constraint relaxation for MPC solves."""

    def __init__(self):
        """Initialize the constraint relaxation handler."""
        self._update_keys = None
        self._relaxation_history = []

    def apply_parameter_updates(
        self,
        instance,
        solver,
        data: Dict[str, Any],
        instance_bound: bool
    ) -> Dict[str, Any]:
        """Apply stochastic parameter updates to the model instance.

        Args:
            instance: Pyomo model instance to update
            solver: Persistent solver instance
            data: Dictionary of parameter updates
            instance_bound: Whether the instance is bound to the solver

        Returns:
            Dictionary of update statistics
        """
        if instance is None:
            logger.info("parameter_update_skipped", reason="instance_not_created")
            return {}

        params_updated = {}

        for key, param in data.items():
            if param["param"]["initialize"] is not None:
                try:
                    param_obj = self._get_parameter_object(instance, key)
                    init_data = param["param"]["initialize"]

                    # Update the parameter value
                    if param["param"]["set"] is not None:
                        self._update_indexed_parameter(param_obj, init_data, param["param"]["set"], key)
                    else:
                        self._update_scalar_parameter(param_obj, init_data)

                    # Track parameter updates
                    params_updated[key] = self._create_update_metadata(param, init_data)

                except (AttributeError, KeyError) as e:
                    raise RuntimeError(
                        f"Failed to update parameter '{key}': {e}. "
                        f"This indicates a structural change which is incompatible with gurobi_persistent. "
                        f"All parameter updates must be in-place (no structural changes)."
                    ) from e

                # Track endogenous parameters for output extraction
                self._track_endogenous_parameter(key, param)

        # Update Gurobi model after parameter changes
        self._update_gurobi_model(solver, instance_bound)

        return params_updated

    def _get_parameter_object(self, instance, key: str):
        """Retrieve parameter object from instance, raising error if not found."""
        param_obj = getattr(instance, key, None)

        if param_obj is None:
            raise RuntimeError(
                f"Parameter '{key}' not found on instance. "
                f"This indicates a structural change which is incompatible with gurobi_persistent."
            )

        return param_obj

    def _update_indexed_parameter(self, param_obj, init_data, param_set, key: str):
        """Update an indexed parameter with new values.

        Args:
            param_obj: Pyomo parameter object
            init_data: New values (dict, list, or tuple)
            param_set: Parameter index set
            key: Parameter name (for error messages)
        """
        if isinstance(init_data, dict):
            # Dict format: keys are indices
            for idx, val in init_data.items():
                if idx in param_obj:
                    param_obj[idx] = val

        elif isinstance(init_data, (list, tuple)):
            # List/tuple format: indices come from the set
            if not isinstance(param_set, (list, tuple)):
                raise RuntimeError(
                    f"Parameter '{key}' has list/tuple initialize but set is not a list/tuple. "
                    f"This indicates a structural change which is incompatible with gurobi_persistent."
                )

            for idx, val in zip(param_set, init_data):
                if idx in param_obj:
                    param_obj[idx] = val
        else:
            raise RuntimeError(
                f"Parameter '{key}' has a set but initialize is not a dict/list/tuple. "
                f"This indicates a structural change which is incompatible with gurobi_persistent."
            )

    def _update_scalar_parameter(self, param_obj, init_data):
        """Update a scalar parameter with a new value.

        Args:
            param_obj: Pyomo parameter object
            init_data: New scalar value
        """
        if hasattr(param_obj, "set_value"):
            param_obj.set_value(init_data)
        else:
            # Scalar parameter, direct assignment
            param_obj.value = init_data

    def _create_update_metadata(self, param: Dict[str, Any], init_data) -> Dict[str, Any]:
        """Create metadata dictionary for a parameter update.

        Args:
            param: Parameter definition dictionary
            init_data: The initialization data

        Returns:
            Dictionary containing update metadata
        """
        signature = self._parameter_signature(init_data)

        return {
            "type": "indexed" if param["param"]["set"] is not None else "scalar",
            "num_values": len(init_data) if isinstance(init_data, (dict, list, tuple)) else 1,
            "signature": signature,
        }

    def _track_endogenous_parameter(self, key: str, param: Dict[str, Any]):
        """Track endogenous parameters for later output extraction.

        Args:
            key: Parameter key
            param: Parameter definition dictionary
        """
        if param["loc"] == "endogenous":
            if self._update_keys is None:
                self._update_keys = {key: param["name"]}
            elif key not in self._update_keys:
                self._update_keys[key] = param["name"]

    def _update_gurobi_model(self, solver, instance_bound: bool):
        """Update the Gurobi model after parameter changes.

        Args:
            solver: Persistent solver instance
            instance_bound: Whether instance is bound to solver
        """
        if instance_bound and hasattr(solver, '_solver_model'):
            try:
                gurobi_model = solver._solver_model
                gurobi_model.update()
                logger.info("gurobi_model_updated_after_params")
            except Exception as e:
                logger.warning("failed_to_update_gurobi_model", error=str(e))

    def apply_start_values(
        self,
        instance,
        solver,
        start_values: Dict,
        instance_bound: bool,
        clean_variable_value_fn
    ) -> Dict[str, int]:
        """Fix variables at t=0 to ensure continuity between MPC solves.

        Args:
            instance: Pyomo model instance
            solver: Persistent solver instance
            start_values: Dictionary of variable values to fix
            instance_bound: Whether instance is bound to solver
            clean_variable_value_fn: Function to clean variable values

        Returns:
            Dictionary with count of fixed variables by name
        """
        fixed_vars = []
        fixed_by_name = {}

        for var in instance.component_objects(Var, active=True):
            if var.name == "cumulative_profit":
                continue
            for index in var:
                key = (var.name, index)
                if key in start_values:
                    cleaned = clean_variable_value_fn(var[index], start_values[key])
                    var[index].fix(cleaned)
                    fixed_vars.append(var[index])

                    # Track for logging
                    if var.name not in fixed_by_name:
                        fixed_by_name[var.name] = 0
                    fixed_by_name[var.name] += 1

        # Notify persistent solver of fixed variables if instance is bound
        if instance_bound and hasattr(solver, 'update_var'):
            for var_obj in fixed_vars:
                solver.update_var(var_obj)

        return fixed_by_name

    def unfix_all_variables(
        self,
        instance,
        solver,
        instance_bound: bool
    ) -> int:
        """Unfix all previously fixed variables before updating parameters.

        Args:
            instance: Pyomo model instance
            solver: Persistent solver instance
            instance_bound: Whether instance is bound to solver

        Returns:
            Count of unfixed variables
        """
        unfixed_vars = []
        for var in instance.component_objects(Var, active=True):
            for index in var:
                if var[index].fixed:
                    var[index].unfix()
                    unfixed_vars.append(var[index])

        # Notify persistent solver of unfixed variables if instance is bound
        if instance_bound and hasattr(solver, 'update_var'):
            for var_obj in unfixed_vars:
                solver.update_var(var_obj)

        return len(unfixed_vars)

    def relax_constraints_on_failure(
        self,
        instance,
        termination_condition: str,
        attempt: int = 0
    ) -> Optional[Dict[str, Any]]:
        """Apply constraint relaxation based on solve failure.

        This method implements a decision tree for progressively relaxing
        constraints when solves fail. Future enhancement will add sophisticated
        relaxation strategies.

        Args:
            instance: Pyomo model instance
            termination_condition: Solver termination condition
            attempt: Current relaxation attempt number

        Returns:
            Dictionary of relaxation parameters applied, or None if no relaxation
        """
        # TODO: Implement decision tree logic for constraint relaxation
        # This is a placeholder for future enhancement

        logger.info(
            "relaxation_decision",
            termination_condition=str(termination_condition),
            attempt=attempt,
            action="no_relaxation_applied"
        )

        # Track relaxation history
        self._relaxation_history.append({
            "termination_condition": str(termination_condition),
            "attempt": attempt,
            "relaxations": None
        })

        return None

    def get_update_keys(self) -> Optional[Dict[str, str]]:
        """Get the update keys tracked during parameter updates."""
        return self._update_keys

    def get_relaxation_history(self) -> list:
        """Get the history of relaxation attempts."""
        return self._relaxation_history

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

class RelaxationTree:
    """Decision tree for progressive constraint relaxation on solve failures."""

    def __init__(self, decision_tree: Optional[Dict[str, Any] | str] = None):
        """Initialize the relaxation tree with a decision tree structure.

        Args:
            decision_tree: Either a dict with decision tree structure,
                          a path to YAML file, or None to use default
        """
        self.decision_tree_input = decision_tree
        self.decision_tree = None
        self._precedence_order = []
        self.relaxation_history = []
        self.last_applied = None
        self.pos = 0
        self._setup()

    def _setup(self):
        """Set up the relaxation tree based on the decision tree structure."""
        import yaml
        from pathlib import Path

        if isinstance(self.decision_tree_input, dict):
            self.decision_tree = self.decision_tree_input
        elif isinstance(self.decision_tree_input, str):
            # Load decision tree from YAML file
            with open(self.decision_tree_input, 'r') as f:
                self.decision_tree = yaml.safe_load(f)
        else:
            # Load default decision tree from src/data/decision_tree.yml
            default_path = Path(__file__).parent.parent / "data" / "decision_tree.yml"
            with open(default_path, 'r') as f:
                self.decision_tree = yaml.safe_load(f)

        # Note: YAML file has typo "presidence_order" instead of "precedence_order"
        self._precedence_order = self.decision_tree.get("presidence_order",
                                                        self.decision_tree.get("precedence_order", []))

    def get_next_relaxation(self, termination_condition: str) -> Optional[Dict[str, Any]]:
        """Get the next relaxation parameters based on current position in tree.

        Args:
            termination_condition: Solver termination condition

        Returns:
            Dictionary of parameter updates, or None if tree exhausted
        """
        if self.pos >= len(self._precedence_order):
            logger.info("relaxation_tree_exhausted",
                       termination_condition=str(termination_condition),
                       attempts=self.pos)
            return None

        # Get the next relaxation step name
        relaxation_name = self._precedence_order[self.pos]

        # Get the parameter modifications for this step
        relaxation_specs = self.decision_tree.get(relaxation_name, [])

        # Parse the relaxation specs (format: ["param_name = value"])
        import numpy as np
        param_updates = {}
        for spec in relaxation_specs:
            if isinstance(spec, str) and "=" in spec:
                param_name, value_str = spec.split("=", 1)
                param_name = param_name.strip()
                value_str = value_str.strip()

                # Convert value string to appropriate type
                try:
                    if value_str.lower() in ("true", "false"):
                        param_updates[param_name] = value_str.lower() == "true"
                    elif value_str.lower() in ("inf", "+inf"):
                        param_updates[param_name] = np.inf
                    elif value_str.lower() == "-inf":
                        param_updates[param_name] = -np.inf
                    elif value_str.lower() == "nan":
                        param_updates[param_name] = np.nan
                    else:
                        param_updates[param_name] = float(value_str) if "." in value_str else int(value_str)
                except ValueError:
                    param_updates[param_name] = value_str

        # Track this relaxation
        self.last_applied = {
            "step": self.pos,
            "name": relaxation_name,
            "params": param_updates,
            "termination_condition": str(termination_condition)
        }
        self.relaxation_history.append(self.last_applied)

        logger.info("relaxation_applied",
                   step=self.pos,
                   relaxation=relaxation_name,
                   params=param_updates,
                   termination_condition=str(termination_condition))

        # Increment position for next call
        self.pos += 1

        return param_updates

    def reset(self):
        """Reset to root of decision tree for next solve iteration."""
        self.pos = 0
        self.last_applied = None
        # Keep history across resets for reporting

    def get_relaxation_count(self) -> int:
        """Get total number of relaxations applied in current iteration."""
        return self.pos

    def get_history(self) -> list:
        """Get full relaxation history."""
        return self.relaxation_history

