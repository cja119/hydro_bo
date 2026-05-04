"""Progressive constraint relaxation for the MPC controller.

`RelaxationTree` walks a YAML-defined decision tree of parameter
overrides on solve failure: each node names a relaxation step
(e.g. "loosen_storage", "drop_charge_limit") whose specs are applied
in precedence order until the solve succeeds or the tree is exhausted.
"""

from typing import Optional, Dict, Any

from hydro_bo.utils.logging_config import get_logger

logger = get_logger(__name__)


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

    def get_relaxable_param_names(self) -> set:
        """Return the set of parameter names this tree may modify.

        Used by the controller to snapshot baseline values so that parameters
        mutated by one relaxed solve do not persist into subsequent solves.
        """
        names = set()
        for relaxation_name in self._precedence_order:
            for spec in self.decision_tree.get(relaxation_name, []):
                if isinstance(spec, str) and "=" in spec:
                    param_name, _ = spec.split("=", 1)
                    names.add(param_name.strip())
        return names
