"""
This isn't really an environment: This class will interact with the planning model (fond at github.com/cja119/stochasticmodel)
and solve the model to perform the planning operationsß for the hydrogen production region.
"""

from __future__ import annotations
from typing import Optional
from h2_plan.opt import H2Planning
from h2_plan.data import DefaultParams, PlanningResults
from pathlib import Path


class Planning:
    def __init__(
        self, uniqe_idx: str, weather_file: str, parameters: Optional[str] = None
    ):
        """
        Initializes the planning model.
        """
        self._weather_file = weather_file
        self._model = None
        self._inputs = None
        self._outputs = None
        self._props = None
        self._idx = uniqe_idx

        if parameters is None:
            self._parameters = DefaultParams("default").formulation_parameters
        else:
            self._parameters = DefaultParams(parameters).formulation_parameters

        self._booleans = {
            "vector_choice": {
                "LH2": False,
                "NH3": False,
            },
            "electrolysers": {"alkaline": False, "PEM": False, "SOFC": False},
            "grid_connection": False,
            "wind": False,
            "solar": False,
            "net_present_value": True,
            "geographical_storage": False,
            "grid_wheel": False,
        }
        self._parameters.update(
            {
                "booleans": self._booleans,
                "stage_duration": 168,
                "n_stages": 3,
                "n_stochastics": 3,
                "hydrogen_price": 5,
                "random_seed": 42,
                "relaxed_ramping": True,
                "vector_operating_duration": 1,
                "shipping_decision": 168,
            }
        )

        return None

    def get_parameters(self) -> str:
        """
        Returns the data of the planning model.
        """
        return self._parameters

    def __enter__(self) -> Planning:
        """
        Enters the planning model context.
        """
        return self._parameters

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """
        Exits the planning model context.
        """

        filepath = Path(__file__).parent.parent.parent / "tmp/planning"

        if not filepath.exists():
            filepath.mkdir(parents=True, exist_ok=True)

        self._model = H2Planning(
            self._parameters, key=self._idx, filename=self._weather_file, filepath=None
        )
        return None

    def save_parameters(self, filename: str, file_path: Optional[str]) -> None:
        """
        Saves the loaded datafile for the planning model
        """
        if file_path is None:
            file_path = self._parameters.path.parent / filename
        else:
            file_path = Path(file_path) / filename

        with open(file_path, "w") as f:
            f.write(self._parameters.to_json())
        return None

    def load_parameters(self, filename: str, file_path: Optional[str]) -> None:
        """
        Loads the data for the planning model.
        """
        if file_path is None:
            file_path = self._parameters.path.parent / filename
        else:
            file_path = Path(file_path) / filename

        with open(file_path, "r") as f:
            self._parameters = DefaultParams.from_json(f.read())
        return None

    def solve(self) -> None:
        """
        Solves the planning model.
        """
        self._model = H2Planning.class_solve(
            key=self._idx, solver="gurobi", verbose=False
        )
        return None

    def get_results(self, target: Optional[str] = None):
        """
        Returns the results of the planning model.
        """
        if target is None:
            target = Path(__file__).parent.parent.parent / "tmp/planning"
        print(target)
        self._res = PlanningResults(self._model)

        return self._res.extract_results(target)

    def visualise(self):
        self._model.generate_plots(self._model)
        return None

    def capex(self) -> dict:
        """
        Returns the capital expenditure of the planning model.
        """
        return self._res.capex()

    def opex(self) -> dict:
        """
        Returns the operational expenditure of the planning model.
        """
        return self._res.opex()
