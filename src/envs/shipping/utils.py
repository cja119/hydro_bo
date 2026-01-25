"""
Utility functions for the shipping environment.
"""

from typing import Optional
from pathlib import Path
import importlib.util
from h2_plan.data import DefaultParams
from pyomo.environ import (
    maximize,
    minimize,
    NonNegativeIntegers,
    NonNegativeReals,
    Reals,
    Binary,
)
from numpy.random import rand
from numpy import ones, mean
from math import floor
from random import randint
from glob import glob
import yaml


def import_mpc_data(
    planning_model: str, vector: str, random_param: bool = False
) -> dict:
    """
    This function is used to import the data from the config file.
    """

    sets = {}
    params = {}
    vars = {}
    forms = {}

    config_file = Path(__file__).parent.parent.parent / "data/shipping" / "config.yml"
    if isinstance(planning_model, str):
        planning_model = (
            Path(__file__).parent.parent.parent / "tmp/planning" / planning_model
        )
    variable_file = (
        Path(__file__).parent.parent.parent / "data/shipping" / "variables.yml"
    )

    default_parameters = DefaultParams("default")

    default_parameters.filter_params(vector)

    default_parameters = default_parameters.formulation_parameters

    if not config_file.exists():
        raise FileNotFoundError(f"Config file {config_file} does not exist.")

    if not planning_model.exists():
        raise FileNotFoundError(f"Planning model {planning_model} does not exist.")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    with open(variable_file, "r") as f:
        variables = yaml.safe_load(f)

    for key, item in config["Time"].items():
        if key != "total_duration":
            sets[key] = [
                val * item for val in range(config["Time"]["total_duration"] // item)
            ]
    for key, item in config["Sets"].items():
        if isinstance(item, list):
            sets[key] = item

    for key, value in config["param_source"]["default_data"].items():
        param = default_parameters.copy()
        for _key in value:
            param = param[_key]

        if isinstance(param, dict):
            for _set in sets:
                if all(_key in param for _key in sets[_set]):
                    params[key] = {
                        _key: param[_key] for _key in sets[_set] if _key in param
                    }

        elif isinstance(param, (list, tuple)):
            if random_param is True:
                param = (param[2] - param[0]) * rand() + param[0]
                params[key] = param
            else:
                params[key] = param[1]
        else:
            params[key] = param

    for item in config["param_source"]["planning_model"]:
        with open(planning_model, "r") as f:
            data = yaml.safe_load(f)

        if item in data:
            params[item] = data[item]

    for _, value in variables["variables"].items():
        vars[value["name"]] = {
            "time_duration": [sets[_key] for _key in value["time_duration"]],
            "domain": (
                NonNegativeReals
                if value["domain"] == "positive_real"
                else (
                    NonNegativeIntegers
                    if value["domain"] == "positive_integer"
                    else (Binary if value["domain"] == "binary" else Reals)
                )
            ),
        }

    for key, param in variables["parameters"].items():
        params[key] = param

    for key, value in config["formulations"].items():
        forms[key] = value
    return {"sets": sets, "params": params, "vars": vars, "forms": forms}


def import_mpc_functions(data_folder: str, sets: dict) -> dict:
    """
    This function is used to import data from the functions file
    """
    funcs = {
        "equations": {},
        "constraints": {},
        "objectives": {},
    }

    equations_path = (
        Path(__file__).parent.parent.parent / "data/shipping" / "equations.py"
    )

    functions_path = (
        Path(__file__).parent.parent.parent / "data/shipping" / "functions.yml"
    )

    if not equations_path.exists():
        raise FileNotFoundError(f"Equations file {equations_path} does not exist.")

    if not functions_path.exists():
        raise FileNotFoundError(f"Functions file {functions_path} does not exist.")

    with open(functions_path, "r") as f:
        functions = yaml.safe_load(f)

    mod_name = equations_path.stem
    spec = importlib.util.spec_from_file_location(mod_name, equations_path)
    _funcs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_funcs)

    for key, value in functions["equations"].items():
        if value["name"] in dir(_funcs):
            func = getattr(_funcs, value["name"])
            funcs["equations"][key] = {
                "time_duration": [sets[_key] for _key in value["domain"]],
                "rule": func,
            }
        else:
            raise ValueError(f"Function {key} not found in equations file.")

    for key, value in functions["constraints"].items():
        if value["name"] in dir(_funcs):
            func = getattr(_funcs, value["name"])
            funcs["constraints"][key] = {
                "time_duration": [sets[_key] for _key in value["domain"]],
                "rule": func,
            }
        else:
            raise ValueError(f"Function {key} not found in constraints file.")

    for key, value in functions["objectives"].items():
        if value["name"] in dir(_funcs):
            func = getattr(_funcs, value["name"])
            funcs["objectives"][key] = {
                "rule": func,
                "sense": maximize if value["sense"] == "max" else minimize,
            }
        else:
            raise ValueError(f"Function {key} not found in objectives file.")
    return funcs


def args_dict():
    args = {
        "vector": None,
        "mpc": {
            "data_folder": None,
            "planning_model": "version2",
            "random_param": False,
            "horizon": 28,
            "solver": "gurobi",
        },
        "slow": {},
        "demand_prediction": {
            "country": "EU",
            "frequency": "monthly",
            "sector": "industry",
            "scale": 24.3,
        },
        "months": 12,
        "weather_data": {"weather_file": None},
        "shipping": {
            "mean_transit_time": 840,
            "std_transit_time": 48,
            "mean_ship_arrival_time": 168,
            "std_ship_arrival_time": 34,
        },
    }
    return args


def temporal_align(weather, randomise: Optional[bool] = False):
    """
    This function temporally aligns the weather data with the demand data.
    and optionaly starts at a random opint in the dataseries.
    """
    if randomise:
        random_start = randint(0, len(weather) - 1)
    else:
        random_start = 0

    weather_data = weather[random_start:] + weather[:random_start]

    return weather_data


def calculate_capex_opex(
    renewables,
    vector,
    compression_capacity,
    electrolyser_capacity,
    fuelcell_capacity,
    conversion_trains_number,
    hydrogen_storage_capacity,
    renewable_energy_capacity,
    vector_storage_capacity,
):

    parameters = DefaultParams().formulation_parameters
    equipment_lives = ones(
        parameters["replacement_frequencies"]["system_duration"] + 20
    )
    for i in range(1, parameters["replacement_frequencies"]["system_duration"]):
        for j in range(1, parameters["replacement_frequencies"]["system_duration"]):
            if j % i == 0:
                equipment_lives[i - 1] += (
                    1 / (1 + parameters["miscillaneous"]["discount_factor"][1])
                ) ** (j - 1)

    _turbine = equipment_lives[
        floor(parameters["replacement_frequencies"]["turbine"] - 1)
    ]
    _solar = equipment_lives[floor(parameters["replacement_frequencies"]["solar"] - 1)]
    _electrolyser = equipment_lives[
        floor(parameters["replacement_frequencies"]["electrolysers"]["SOFC"][1] - 1)
    ]
    _fuel_cell = equipment_lives[
        floor(parameters["replacement_frequencies"]["fuel_cell"] - 1)
    ]
    _h2storage = equipment_lives[
        floor(parameters["replacement_frequencies"]["hydrogen_storage"] - 1)
    ]
    _NH3storage = equipment_lives[
        floor(parameters["replacement_frequencies"]["vector_storage"]["NH3"] - 1)
    ]
    _LH2storage = equipment_lives[
        floor(parameters["replacement_frequencies"]["vector_storage"]["LH2"] - 1)
    ]
    _compression = equipment_lives[
        floor((parameters["replacement_frequencies"]["compressor"][1]) - 1)
    ]
    _LH2prod = equipment_lives[
        floor(parameters["replacement_frequencies"]["vector_production"]["LH2"] - 1)
    ]
    _NH3prod = equipment_lives[
        floor(parameters["replacement_frequencies"]["vector_production"]["NH3"] - 1)
    ]
    _plant = equipment_lives[0]

    capex = {}
    opex = {}

    capex["renewables"] = renewable_energy_capacity * (
        parameters["capital_costs"]["turbine"][1] * _turbine
        if renewables == "wind"
        else parameters["capital_costs"]["solar"][1] * _solar
    )
    opex["renewables"] = (
        renewable_energy_capacity
        * (
            parameters["operating_costs"]["turbine"][1]
            if renewables == "wind"
            else parameters["operating_costs"]["solar"][1]
        )
        * _plant
    )

    capex["electrolyser"] = (
        electrolyser_capacity
        * _electrolyser
        * parameters["capital_costs"]["electrolysers"]["SOFC"][1]
    )
    opex["electrolyser"] = (
        electrolyser_capacity
        * parameters["operating_costs"]["electrolysers"]["SOFC"][1]
        * _plant
    )

    capex["fuel_cell"] = (
        fuelcell_capacity * _fuel_cell * parameters["capital_costs"]["fuel_cell"][1]
    )
    opex["fuel_cell"] = (
        fuelcell_capacity * parameters["operating_costs"]["fuel_cell"][1] * _plant
    )

    capex["hydrogen_storage"] = (
        hydrogen_storage_capacity
        * _h2storage
        * parameters["capital_costs"]["hydrogen_storage"][1]
        / 120
    )
    opex["hydrogen_storage"] = (
        hydrogen_storage_capacity
        * parameters["operating_costs"]["hydrogen_storage"][1]
        * _plant
        / 120
    )

    if vector == "NH3":
        capex["vector_storage"] = (
            vector_storage_capacity
            * _NH3storage
            * parameters["capital_costs"]["vector_storage"]["NH3"][1]
        )
        opex["vector_storage"] = (
            vector_storage_capacity
            * parameters["operating_costs"]["vector_storage"]["NH3"][1]
            * _plant
        )
        capex["vector_production"] = (
            _NH3prod
            * parameters["capital_costs"]["vector_production"]["NH3"][1]
            * (parameters["vector_production"]["single_train_throughput"]["NH3"])
            ** (2 / 3)
            * conversion_trains_number
        )
        opex["vector_production"] = (
            parameters["operating_costs"]["vector_production"]["NH3"][1]
            * (parameters["vector_production"]["single_train_throughput"]["NH3"])
            * _plant
            * conversion_trains_number
        )

    elif vector == "LH2":
        capex["vector_storage"] = (
            vector_storage_capacity
            * _LH2storage
            * parameters["capital_costs"]["vector_storage"]["LH2"][1]
        )
        opex["vector_storage"] = (
            vector_storage_capacity
            * parameters["operating_costs"]["vector_storage"]["LH2"][1]
            * _plant
        )
        capex["vector_production"] = (
            _LH2prod
            * parameters["capital_costs"]["vector_production"]["LH2"][1]
            * (parameters["vector_production"]["single_train_throughput"]["LH2"][1])
            ** (2 / 3)
            * conversion_trains_number
        )
        opex["vector_production"] = (
            parameters["operating_costs"]["vector_production"]["LH2"][1]
            * (parameters["vector_production"]["single_train_throughput"]["LH2"][1])
            * _plant
            * conversion_trains_number
        )

    capex["compression"] = (
        compression_capacity
        * _compression
        * parameters["capital_costs"]["compressor"]
        / 120
    )
    opex["compression"] = (
        compression_capacity
        * parameters["operating_costs"]["compressor"]
        * _plant
        / 120
    )
    total_capex = float(sum(capex.values()))
    total_opex = float(sum(opex.values()))
    results = {
        "capex": total_capex,
        "compression_capacity": compression_capacity,
        "conversion_trains_number": conversion_trains_number,
        "electrolyser_capacity": electrolyser_capacity,
        "fuelcell_capacity": fuelcell_capacity,
        "hydrogen_storage_capacity": hydrogen_storage_capacity,
        "opex": total_opex,
        "renewable_energy_capacity": renewable_energy_capacity,
        "renewables": renewables,
        "vector": vector,
        "vector_storage_capacity": vector_storage_capacity,
    }

    return results


def generate_weather_forecast(weather_data, start_idx, grid_set):
    end = start_idx + 168
    full_length = len(grid_set)
    mean_weather = mean(weather_data)

    forecast = weather_data[start_idx:end]
    if len(forecast) < full_length:
        forecast += (full_length - len(forecast)) * [mean_weather]
    return forecast


def count_and_shift_arrivals(ship_origin, expected_arrivals, size):
    arrived = ship_origin[size].count(0)
    ship_origin[size][:] = [x for x in ship_origin[size] if x > 0]
    expected_arrivals[size][:] = expected_arrivals[size][arrived:]
    return arrived


def status_message(day: int, month: int) -> None:
    print(
        f"\r[Inner-Loop] Shipping schedule for day {day}, month {month}.",
        end=" " * 20,
    )
    return None
