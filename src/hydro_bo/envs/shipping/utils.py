"""
Utility functions for the shipping environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple
from pathlib import Path
import importlib.util
import matplotlib.pyplot as plt
import numpy as np
from h2_plan.data import DefaultParams
from pyomo.environ import (
    maximize,
    minimize,
    NonNegativeIntegers,
    NonNegativeReals,
    Reals,
    Binary,
)
from numpy import ones, mean
from math import floor
import yaml

from hydro_bo.utils.seeding import make_rng


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
TMP_DIR = PROJECT_ROOT / "tmp"
EQUATIONS_PATH = Path(__file__).parent / "equations.py"
FUNCTIONS_PATH = DATA_DIR / "functions.yml"
CONFIG_PATH = DATA_DIR / "config.yml"
VARIABLES_PATH = DATA_DIR / "variables.yml"


def _load_yaml(path: Path) -> MutableMapping[str, Any]:
    """Load a YAML file, ensuring it exists first."""

    if not path.exists():
        raise FileNotFoundError(f"Required file {path} does not exist.")

    with open(path, "r") as f:
        return yaml.safe_load(f)


def _resolve_planning_model(planning_model: str | Path) -> Path:
    """Normalise planning model input to an on-disk path."""

    candidate = Path(planning_model)
    if candidate.is_absolute():
        return candidate

    if isinstance(planning_model, Path):
        return planning_model

    return Path.cwd() / "tmp" / "planning" / planning_model


def _traverse(mapping: Mapping[str, Any], keys: Iterable[str]) -> Any:
    """Walk nested mappings following the provided keys."""

    current: Any = mapping
    for key in keys:
        current = current[key]
    return current


def import_mpc_data(
    planning_model: str | Path | dict, vector: str, random_param: bool = False,
    random_param_seed: int = 0,
) -> Dict[str, Dict[str, Any]]:
    """Import MPC input data from config, variables, and planning model files."""

    sets: Dict[str, list[Any]] = {}
    params: Dict[str, Any] = {}
    vars: Dict[str, Any] = {}
    forms: Dict[str, Any] = {}

    config = _load_yaml(CONFIG_PATH)
    variables = _load_yaml(VARIABLES_PATH)

    planning_model_path = None
    if isinstance(planning_model, dict):
        planning_model_data = planning_model
    else:
        planning_model_path = _resolve_planning_model(planning_model)

        if not planning_model_path.exists():
            raise FileNotFoundError(f"Planning model {planning_model_path} does not exist.")

    default_parameters = DefaultParams("default")
    default_parameters.filter_params(vector)
    parameter_tree = default_parameters.formulation_parameters

    time_config = config.get("Time", {})
    total_duration = time_config.get("total_duration")

    if total_duration is None:
        raise KeyError("Time.total_duration is required in config.yml")

    # `forecast_horizon` is a scalar, not a grid spacing — keep it out of
    # the set-construction loop and pull it out for validation below.
    forecast_horizon = int(time_config.get("forecast_horizon", 168))
    if forecast_horizon < 24:
        raise ValueError(
            f"Time.forecast_horizon must be >= 24h (got {forecast_horizon}); "
            "anything shorter starves the MPC of intra-day signal."
        )
    if forecast_horizon > int(total_duration):
        raise ValueError(
            f"Time.forecast_horizon ({forecast_horizon}) cannot exceed "
            f"Time.total_duration ({int(total_duration)})."
        )

    for key, step in time_config.items():
        if key in ("total_duration", "forecast_horizon"):
            continue
        sets[key] = [val * step for val in range(total_duration // step)]

    for key, item in config.get("Sets", {}).items():
        if isinstance(item, list):
            sets[key] = item

    for key, path in config.get("param_source", {}).get("default_data", {}).items():
        param = _traverse(parameter_tree, path)

        if isinstance(param, dict):
            for set_name, members in sets.items():
                if all(member in param for member in members):
                    params[key] = {member: param[member] for member in members if member in param}

        elif isinstance(param, (list, tuple)):
            if random_param:
                if "_random_param_rng" not in locals():
                    _random_param_rng = make_rng(random_param_seed)
                params[key] = (param[2] - param[0]) * float(_random_param_rng.random()) + param[0]
            else:
                params[key] = param[1]
        else:
            params[key] = param

    # variables.yml defaults run BEFORE planning_model overrides so the
    # BO can supply values (e.g. storage backoffs) that override the
    # defaults via `param_overrides` rather than being clobbered by them.
    for key, param in variables.get("parameters", {}).items():
        params[key] = param

    if planning_model_path is not None:
        planning_model_data = _load_yaml(planning_model_path)
    for item in config.get("param_source", {}).get("planning_model", []):
        if item in planning_model_data:
            params[item] = planning_model_data[item]

    domain_map = {
        "positive_real": NonNegativeReals,
        "positive_integer": NonNegativeIntegers,
        "binary": Binary,
    }

    for value in variables.get("variables", {}).values():
        vars[value["name"]] = {
            "time_duration": [sets[_key] for _key in value["time_duration"]],
            "domain": domain_map.get(value["domain"], Reals),
        }

    for key, value in config.get("formulations", {}).items():
        forms[key] = value

    price_dynamics = config.get("price_dynamics", {"enabled": False})

    # Clamp the MPC's assumed-arrival shift so (mean + offset) stays in
    # [1, max grid1 day]. Out-of-range values would produce negative or
    # over-horizon time indices in port_capacity / ship_schedule_aux.
    mat = int(params.get("mean_ship_arrival_time", 7))
    off = int(params.get("expected_arrival_offset", 0))
    max_day = max(1, int(total_duration) // 24 - 1)
    effective = max(1, min(max_day, mat + off))
    params["expected_arrival_offset"] = effective - mat

    return {
        "sets": sets,
        "params": params,
        "vars": vars,
        "forms": forms,
        "price_dynamics": price_dynamics,
        "forecast_horizon": forecast_horizon,
    }


def import_mpc_functions(data_folder: str, sets: Mapping[str, Sequence[Any]]) -> Dict[str, Dict[str, Any]]:
    """Import callable MPC components defined in equations.py and functions.yml."""

    _ = data_folder  # placeholder for future use; keeps API stable

    funcs: Dict[str, Dict[str, Any]] = {"equations": {}, "constraints": {}, "objectives": {}}

    if not EQUATIONS_PATH.exists():
        raise FileNotFoundError(f"Equations file {EQUATIONS_PATH} does not exist.")

    if not FUNCTIONS_PATH.exists():
        raise FileNotFoundError(f"Functions file {FUNCTIONS_PATH} does not exist.")

    functions = _load_yaml(FUNCTIONS_PATH)

    mod_name = EQUATIONS_PATH.stem
    spec = importlib.util.spec_from_file_location(mod_name, EQUATIONS_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module specification from {EQUATIONS_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[operator]

    for key, value in functions.get("equations", {}).items():
        if value["name"] not in dir(module):
            raise ValueError(f"Function {key} not found in equations file.")
        func = getattr(module, value["name"])
        funcs["equations"][key] = {
            "time_duration": [sets[_key] for _key in value["domain"]],
            "rule": func,
        }

    for key, value in functions.get("constraints", {}).items():
        if value["name"] not in dir(module):
            raise ValueError(f"Function {key} not found in constraints file.")
        func = getattr(module, value["name"])
        funcs["constraints"][key] = {
            "time_duration": [sets[_key] for _key in value["domain"]],
            "rule": func,
        }

    for key, value in functions.get("objectives", {}).items():
        if value["name"] not in dir(module):
            raise ValueError(f"Function {key} not found in objectives file.")
        func = getattr(module, value["name"])
        funcs["objectives"][key] = {
            "rule": func,
            "sense": maximize if value["sense"] == "max" else minimize,
        }

    return funcs


def args_dict() -> Dict[str, Any]:
    """Return default argument structure for shipping/planning runs."""

    return {
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
        "seed": 42,
        "price_dynamics": {"enabled": False},
    }


def temporal_align(weather: Sequence[Any], randomise: bool = False, seed: int = 42) -> list[Any]:
    """Align weather data to a random or zero offset."""
    if randomise:
        rng = make_rng(seed)
        random_start = int(rng.integers(0, len(weather)))
    else:
        random_start = 0
    return list(weather[random_start:]) + list(weather[:random_start])


def calculate_capex_opex(
    renewables: str,
    vector: str,
    compression_capacity: float,
    electrolyser_capacity: float,
    fuelcell_capacity: float,
    conversion_trains_number: int,
    hydrogen_storage_capacity: float,
    renewable_energy_capacity: float,
    vector_storage_capacity: float,
) -> Dict[str, float | int | str]:
    """Compute CAPEX and OPEX based on capacity choices."""

    parameters = DefaultParams().formulation_parameters
    system_duration = parameters["replacement_frequencies"]["system_duration"]
    discount_factor = parameters["miscillaneous"]["discount_factor"][1]

    equipment_lives = ones(system_duration + 20)
    for interval in range(1, system_duration):
        for year in range(interval, system_duration):
            if year % interval == 0:
                equipment_lives[interval - 1] += (1 / (1 + discount_factor)) ** (year - 1)

    _turbine = equipment_lives[floor(parameters["replacement_frequencies"]["turbine"] - 1)]
    _solar = equipment_lives[floor(parameters["replacement_frequencies"]["solar"] - 1)]
    _electrolyser = equipment_lives[floor(parameters["replacement_frequencies"]["electrolysers"]["SOFC"][1] - 1)]
    _fuel_cell = equipment_lives[floor(parameters["replacement_frequencies"]["fuel_cell"] - 1)]
    _h2storage = equipment_lives[floor(parameters["replacement_frequencies"]["hydrogen_storage"] - 1)]
    _NH3storage = equipment_lives[floor(parameters["replacement_frequencies"]["vector_storage"]["NH3"] - 1)]
    _LH2storage = equipment_lives[floor(parameters["replacement_frequencies"]["vector_storage"]["LH2"] - 1)]
    _compression = equipment_lives[floor((parameters["replacement_frequencies"]["compressor"][1]) - 1)]
    _LH2prod = equipment_lives[floor(parameters["replacement_frequencies"]["vector_production"]["LH2"] - 1)]
    _NH3prod = equipment_lives[floor(parameters["replacement_frequencies"]["vector_production"]["NH3"] - 1)]
    _plant = equipment_lives[0]

    capex: Dict[str, float] = {}
    opex: Dict[str, float] = {}

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

    capex["electrolyser"] = electrolyser_capacity * _electrolyser * parameters["capital_costs"]["electrolysers"]["SOFC"][1]
    opex["electrolyser"] = electrolyser_capacity * parameters["operating_costs"]["electrolysers"]["SOFC"][1] * _plant

    capex["fuel_cell"] = fuelcell_capacity * _fuel_cell * parameters["capital_costs"]["fuel_cell"][1]
    opex["fuel_cell"] = fuelcell_capacity * parameters["operating_costs"]["fuel_cell"][1] * _plant

    capex["hydrogen_storage"] = hydrogen_storage_capacity * _h2storage * parameters["capital_costs"]["hydrogen_storage"][1] / 120
    opex["hydrogen_storage"] = hydrogen_storage_capacity * parameters["operating_costs"]["hydrogen_storage"][1] * _plant / 120

    if vector == "NH3":
        capex["vector_storage"] = vector_storage_capacity * _NH3storage * parameters["capital_costs"]["vector_storage"]["NH3"][1]
        opex["vector_storage"] = vector_storage_capacity * parameters["operating_costs"]["vector_storage"]["NH3"][1] * _plant
        capex["vector_production"] = (
            _NH3prod
            * parameters["capital_costs"]["vector_production"]["NH3"][1]
            * (parameters["vector_production"]["single_train_throughput"]["NH3"])
            ** (2 / 3)
            * conversion_trains_number
        )
        opex["vector_production"] = (
            parameters["operating_costs"]["vector_production"]["NH3"][1]
            * parameters["vector_production"]["single_train_throughput"]["NH3"] ** (2 / 3)
            * _plant
            * conversion_trains_number
        )

    elif vector == "LH2":
        capex["vector_storage"] = vector_storage_capacity * _LH2storage * parameters["capital_costs"]["vector_storage"]["LH2"][1]
        opex["vector_storage"] = vector_storage_capacity * parameters["operating_costs"]["vector_storage"]["LH2"][1] * _plant
        capex["vector_production"] = (
            _LH2prod
            * parameters["capital_costs"]["vector_production"]["LH2"][1]
            * (parameters["vector_production"]["single_train_throughput"]["LH2"][1])
            ** (2 / 3)
            * conversion_trains_number
        )
        opex["vector_production"] = (
            parameters["operating_costs"]["vector_production"]["LH2"][1]
            * parameters["vector_production"]["single_train_throughput"]["LH2"][1] ** (2 / 3)
            * _plant
            * conversion_trains_number
        )

    capex["compression"] = compression_capacity * _compression * parameters["capital_costs"]["compressor"] / 120
    opex["compression"] = compression_capacity * parameters["operating_costs"]["compressor"] * _plant / 120

    total_capex = float(sum(capex.values()))
    total_opex = float(sum(opex.values()))

    return {
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


WIND_FORECAST_SCALE = 11.88


def generate_weather_forecast(
    weather_data: Sequence[float],
    start_idx: int,
    grid_set: Sequence[Any],
    forecast_horizon: int = 168,
    mean_override: Optional[float] = None,
) -> list[float]:
    """Slice `forecast_horizon` hours of true weather starting at
    `start_idx`, then pad to the full grid length with the climatological
    mean. The horizon is the design knob that controls how much
    look-ahead the MPC actually receives — anything beyond it is just
    flat mean weather. Caller is expected to validate the horizon
    against `total_duration`; we still clip defensively against the
    grid length to avoid an empty slice.

    `mean_override` (in [0, 1]) replaces the horizon slice with a flat
    value `mean_override * WIND_FORECAST_SCALE`, projecting the BO knob
    into the same units the MPC expects. The post-horizon tail still
    pads with the climatological mean of the true series."""
    full_length = len(grid_set)
    horizon = max(1, min(int(forecast_horizon), full_length))
    end = start_idx + horizon
    mean_weather = float(mean(weather_data))

    if mean_override is not None:
        forecast = horizon * [float(mean_override) * WIND_FORECAST_SCALE]
    else:
        forecast = list(weather_data[start_idx:end])
    if len(forecast) < full_length:
        forecast += (full_length - len(forecast)) * [mean_weather]
    return forecast


def count_and_shift_arrivals(
    ship_origin: MutableMapping[int, list[int]],
    expected_arrivals: MutableMapping[int, list[int]],
    size: int,
    ship_origin_latent: MutableMapping[int, list[float]] | None = None,
    ship_origin_baseline: MutableMapping[int, list[int]] | None = None,
) -> int:
    origin = ship_origin[size]
    expected = expected_arrivals[size]

    # Keep actual and expected lists index-aligned so a realised arrival removes
    # the matching expected ship. This avoids dropping/duplicating expectations.
    if len(origin) == len(expected):
        keep_idx = [i for i, remaining in enumerate(origin) if remaining > 0]
        arrived = len(origin) - len(keep_idx)
        ship_origin[size][:] = [origin[i] for i in keep_idx]
        expected_arrivals[size][:] = [expected[i] for i in keep_idx]
        if ship_origin_latent is not None and size in ship_origin_latent:
            latent = ship_origin_latent[size]
            ship_origin_latent[size][:] = [latent[i] for i in keep_idx]
        if ship_origin_baseline is not None and size in ship_origin_baseline:
            baseline = ship_origin_baseline[size]
            ship_origin_baseline[size][:] = [baseline[i] for i in keep_idx]
        return arrived

    # Fallback for legacy/misaligned state: any nonpositive ETA means arrived.
    arrived = sum(1 for remaining in origin if remaining <= 0)
    ship_origin[size][:] = [x for x in origin if x > 0]
    expected_arrivals[size][:] = expected[arrived:]
    if ship_origin_latent is not None and size in ship_origin_latent:
        ship_origin_latent[size][:] = ship_origin_latent[size][arrived:]
    if ship_origin_baseline is not None and size in ship_origin_baseline:
        ship_origin_baseline[size][:] = ship_origin_baseline[size][arrived:]
    return arrived


def status_message(day: int, month: int) -> None:
    print(
        f"\r[Inner-Loop] Shipping schedule for day {day}, month {month}.",
        end=" " * 20,
    )


# ---------------------------------------------------------------------------
# Plotting utilities for ShippingEnv.plot_data() outputs
# ---------------------------------------------------------------------------

BURGUNDY = "b"
NAVY = "r"
TITLE_PAD = 10


@dataclass(frozen=True)
class PlotTheme:
    font_size: int = 16
    title_pad: int = TITLE_PAD
    grid_alpha: float = 0.25
    lw: float = 2.0

    ships_ylim: Tuple[float, float] = (0.0, 2.0)
    scatter_s: float = 14.0
    scatter_alpha: float = 0.25


def apply_theme(theme: PlotTheme) -> None:
    plt.rcParams.update({"font.size": theme.font_size})


def _require_keys(data: Mapping[str, Any], keys: Sequence[str], *, where: str) -> None:
    missing = [k for k in keys if k not in data]
    if missing:
        raise KeyError(f"{where}: joined_data missing required keys: {missing}")


def _as_1d_float_array(x: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D after coercion; got shape {arr.shape}")
    return arr


def _as_list(x: Any, *, name: str) -> List[Any]:
    if isinstance(x, list):
        return x
    if isinstance(x, (tuple, np.ndarray)):
        return list(x)
    try:
        return list(x)
    except TypeError as e:
        raise TypeError(f"{name} must be iterable; got {type(x)!r}") from e


def _coerce_joined_data_types(joined_data: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    out = dict(joined_data)

    if "steps" in out:
        out["steps"] = _as_list(_as_1d_float_array(out["steps"], name="steps"), name="steps")
    if "daily_steps" in out:
        out["daily_steps"] = _as_list(_as_1d_float_array(out["daily_steps"], name="daily_steps"), name="daily_steps")

    seq_keys = [
        "vector_storage",
        "cumulative_charge",
        "vector_flux",
        "energy_turbine",
        "energy_fuelcell",
        "hydrogen_storage",
        "n_active_trains_conversion",
    ]
    for k in seq_keys:
        if k in out:
            out[k] = _as_list(_as_1d_float_array(out[k], name=k), name=k)

    dict_seq_keys = ["n_ordered", "n_ship_sent"]
    for dk in dict_seq_keys:
        if dk in out and isinstance(out[dk], Mapping):
            coerced: Dict[Any, List[float]] = {}
            for ship_id, series in out[dk].items():
                coerced[ship_id] = _as_list(
                    _as_1d_float_array(series, name=f"{dk}[{ship_id}]"),
                    name=f"{dk}[{ship_id}]",
                )
            out[dk] = coerced

    return out


def _ensure_same_length(x: Sequence[float], y: Sequence[float], *, name_x: str, name_y: str) -> None:
    if len(x) != len(y):
        raise ValueError(f"Length mismatch: len({name_x})={len(x)} vs len({name_y})={len(y)}")


def _ensure_all_same_length(series: Iterable[Sequence[float]], *, expected: int, name: str) -> None:
    for i, s in enumerate(series):
        if len(s) != expected:
            raise ValueError(f"{name}: series #{i} has length {len(s)}; expected {expected}")


def _fmt_total(v: float) -> str:
    if abs(v) >= 1e6:
        return f"{v/1e6:.2f}M"
    if abs(v) >= 1e3:
        return f"{v/1e3:.2f}k"
    if abs(v - round(v)) < 1e-9:
        return f"{int(round(v))}"
    return f"{v:.2f}"


def _build_ship_bins_from_keys(ship_keys: Sequence[Any]) -> Dict[str, List[Any]]:
    bins: Dict[str, List[Any]] = {"Small": [], "Medium": [], "Large": []}
    unknown: List[Any] = []

    for k in ship_keys:
        s = str(k).lower()
        if "small" in s:
            bins["Small"].append(k)
        elif "large" in s:
            bins["Large"].append(k)
        elif "medium" in s:
            bins["Medium"].append(k)
        else:
            unknown.append(k)

    if unknown:
        raise ValueError(
            "Ship identifiers must include size tags 'Small', 'Medium', or 'Large'; "
            f"found incompatible keys: {unknown}"
        )

    return bins


def _set_max_ticks(ax, xs: Sequence[float], max_ticks: Optional[int]) -> None:
    if max_ticks is None or max_ticks <= 0:
        return
    if len(xs) > max_ticks:
        xticks = list(np.linspace(xs[0], xs[-1], max_ticks))
        ax.set_xticks(xticks)


def plot_style(
    ax,
    x: Sequence[float],
    y: Sequence[float],
    *,
    title: str,
    ylabel: str,
    line_color: str = BURGUNDY,
):
    x_arr = _as_1d_float_array(x, name="x")
    y_arr = _as_1d_float_array(y, name="y")
    _ensure_same_length(x_arr, y_arr, name_x="x", name_y="y")

    ax.plot(x_arr, y_arr, color=line_color, linewidth=1.8, zorder=3)

    ax.set_title(title, fontsize=9, loc="left", pad=2)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    return ax


def plot_scheduling_figure(
    joined_data: Mapping[str, Any],
    *,
    figsize: Tuple[float, float] = (12, 14),
    max_ticks: int = 10,
    theme: PlotTheme = PlotTheme(),
):
    required = [
        "steps",
        "daily_steps",
        "n_ordered",
        "n_ship_sent",
        "vector_storage",
        "cumulative_charge",
    ]
    _require_keys(joined_data, required, where="plot_scheduling_figure")

    jd = _coerce_joined_data_types(dict(joined_data))
    x_daily = _as_1d_float_array(jd["daily_steps"], name="daily_steps")
    x_steps = _as_1d_float_array(jd["steps"], name="steps")

    n_ordered: Mapping[Any, Sequence[float]] = jd["n_ordered"]
    n_sent: Mapping[Any, Sequence[float]] = jd["n_ship_sent"]
    if not isinstance(n_ordered, Mapping) or not isinstance(n_sent, Mapping):
        raise TypeError("n_ordered and n_ship_sent must be mappings of ship_id -> series")

    ship_ids = list(n_ordered.keys())
    bins = _build_ship_bins_from_keys(ship_ids)

    L = len(x_daily)
    if L < 2:
        raise ValueError("daily_steps must have at least 2 points (needed for step/scatter midpoints)")

    _ensure_all_same_length([n_ordered[s] for s in ship_ids], expected=L, name="n_ordered")
    _ensure_all_same_length([n_sent[s] for s in ship_ids], expected=L, name="n_ship_sent")

    fig, axs = plt.subplots(3, 1, figsize=figsize)
    axs = np.asarray(axs, dtype=object).ravel()
    ax_ships, ax_vec, ax_fill = axs

    size_marker = {"Small": "o", "Medium": "^", "Large": "s"}
    ship_marker_size = {"Small": 95, "Medium": 95, "Large": 95}

    ordered_alpha = {"Small": 0.65, "Medium": 0.65, "Large": 0.65}
    sent_alpha = {"Small": 0.65, "Medium": 0.65, "Large": 0.65}
    ordered_color = {"Small": "#7fbfff", "Medium": "b", "Large": "#003d99"}
    sent_color = {"Small": "#ff9999", "Medium": "r", "Large": "#990000"}

    x_mid = (x_daily[:-1] + x_daily[1:]) / 2.0
    totals = {sz: {"ordered": 0.0, "sent": 0.0} for sz in ["Small", "Medium", "Large"]}

    for size_name, ships_in_bin in bins.items():
        if not ships_in_bin:
            continue

        y_ord = np.zeros(L, dtype=float)
        y_snt = np.zeros(L, dtype=float)
        for s in ships_in_bin:
            y_ord += _as_1d_float_array(n_ordered[s], name=f"n_ordered[{s}]")
            y_snt += _as_1d_float_array(n_sent[s], name=f"n_ship_sent[{s}]")

        totals[size_name]["ordered"] = float(np.sum(y_ord))
        totals[size_name]["sent"] = float(np.sum(y_snt))

        ax_ships.step(
            x_daily,
            y_ord,
            where="post",
            color=ordered_color[size_name],
            linewidth=theme.lw * 0.5,
            alpha=ordered_alpha[size_name],
            zorder=3,
        )
        ax_ships.step(
            x_daily,
            y_snt,
            where="post",
            color=sent_color[size_name],
            linestyle="--",
            linewidth=theme.lw * 0.5,
            alpha=sent_alpha[size_name],
            zorder=3,
        )

        y_ord_mid = y_ord[:-1]
        y_snt_mid = y_snt[:-1]
        m_ord = y_ord_mid > 1e-3
        m_snt = y_snt_mid > 1e-3

        ax_ships.scatter(
            x_mid[m_ord],
            y_ord_mid[m_ord],
            color=ordered_color[size_name],
            s=ship_marker_size[size_name] * 0.3,
            alpha=min(1.0, theme.scatter_alpha * 2),
            marker=size_marker[size_name],
            edgecolors="none",
            zorder=4,
        )
        ax_ships.scatter(
            x_mid[m_snt],
            y_snt_mid[m_snt],
            color=sent_color[size_name],
            s=ship_marker_size[size_name] * 0.3,
            alpha=min(1.0, theme.scatter_alpha * 2),
            marker=size_marker[size_name],
            edgecolors="none",
            zorder=4,
        )

    present_sizes = [sz for sz in ["Small", "Medium", "Large"] if bins.get(sz)]
    ordered_handles: List[Any] = []
    sent_handles: List[Any] = []

    for size_name in present_sizes:
        ordered_handles.append(
            plt.Line2D(
                [0],
                [0],
                color=ordered_color[size_name],
                linestyle="-",
                linewidth=theme.lw * 0.3,
                marker=size_marker[size_name],
                markersize=float(np.sqrt(ship_marker_size[size_name]) / 2),
                markerfacecolor=ordered_color[size_name],
                markeredgewidth=0,
                alpha=ordered_alpha[size_name],
                label=f"Ordered ({size_name})  Σ={_fmt_total(totals[size_name]['ordered'])}",
            )
        )
        sent_handles.append(
            plt.Line2D(
                [0],
                [0],
                color=sent_color[size_name],
                linestyle="--",
                linewidth=theme.lw * 0.3,
                marker=size_marker[size_name],
                markersize=float(np.sqrt(ship_marker_size[size_name]) / 2),
                markerfacecolor=sent_color[size_name],
                markeredgewidth=0,
                alpha=sent_alpha[size_name],
                label=f"Sent ({size_name})  Σ={_fmt_total(totals[size_name]['sent'])}",
            )
        )

    if ordered_handles or sent_handles:
        ax_ships.legend(
            handles=ordered_handles + sent_handles,
            loc="best",
            fontsize=7,
            ncols=2,
            frameon=False,
            handlelength=2.2,
            handletextpad=0.6,
            columnspacing=1.2,
        )

    ax_ships.set_title("Ships Ordered / Sent", fontsize=9, loc="left", pad=2)
    ax_ships.set_ylabel("Count", fontsize=9)
    ax_ships.grid(True, linewidth=0.4, alpha=0.5)
    ax_ships.set_ylim(theme.ships_ylim)

    plot_style(
        ax_vec,
        jd["steps"],
        jd["vector_storage"],
        title="Stored Vector",
        ylabel="Mass Stored [kt]",
        line_color=BURGUNDY,
    )

    ship_fill_kt = [float(v) / 1000.0 for v in jd["cumulative_charge"]]
    plot_style(
        ax_fill,
        jd["steps"],
        ship_fill_kt,
        title="Ship Fill",
        ylabel="Mass on Ships [kt]",
        line_color=BURGUNDY,
    )

    ax_fill.set_xlabel("Time [h]", fontsize=9)
    for ax in (ax_ships, ax_vec):
        ax.tick_params(labelbottom=False)
    _set_max_ticks(ax_fill, x_steps, max_ticks)

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.subplots_adjust(hspace=0.25)
    fig.canvas.draw()
    return fig, axs


def plot_energy_management_figure(
    joined_data: Mapping[str, Any],
    *,
    figsize: Tuple[float, float] = (12, 12),
    max_ticks: int = 10,
):
    required = [
        "steps",
        "n_active_trains_conversion",
        "vector_flux",
        "energy_turbine",
        "energy_fuelcell",
        "hydrogen_storage",
    ]
    _require_keys(joined_data, required, where="plot_energy_management_figure")

    jd = _coerce_joined_data_types(dict(joined_data))
    x_steps = _as_1d_float_array(jd["steps"], name="steps")

    fig, axs = plt.subplots(4, 1, figsize=figsize)
    axs = np.asarray(axs, dtype=object).ravel()
    ax_trains, ax_turb, ax_fc, ax_h2 = axs

    ax_trains_r = ax_trains.twinx()

    y_flux = _as_1d_float_array(jd["vector_flux"], name="vector_flux")
    _ensure_same_length(x_steps, y_flux, name_x="steps", name_y="vector_flux")

    ax_trains_r.plot(x_steps, y_flux, color=NAVY, linewidth=1.8, zorder=3)
    ax_trains_r.set_ylabel("Vector Flux [GJ/h]", fontsize=9)

    y_trains = _as_1d_float_array(jd["n_active_trains_conversion"], name="n_active_trains_conversion")
    _ensure_same_length(x_steps, y_trains, name_x="steps", name_y="n_active_trains_conversion")

    ax_trains.plot(x_steps, y_trains, color=BURGUNDY, linewidth=1.8, zorder=3)

    ax_trains.set_title("Conversion Process Throughput", fontsize=9, loc="left", pad=2)
    ax_trains.set_ylabel("Active Trains [count]", fontsize=9)
    ax_trains.grid(True, linewidth=0.4, alpha=0.5)

    h1 = plt.Line2D(
        [0],
        [0],
        color=BURGUNDY,
        linestyle="-",
        linewidth=1.8,
        label="Number Active Trains",
    )
    h2 = plt.Line2D(
        [0],
        [0],
        color=NAVY,
        linestyle="-",
        linewidth=1.8,
        label="Vector Flux",
    )
    ax_trains.legend(handles=[h1, h2], loc="best", fontsize=7, frameon=False)

    plot_style(
        ax_turb,
        jd["steps"],
        jd["energy_turbine"],
        title="Single Turbine Energy",
        ylabel="Energy [GJ/h]",
        line_color=BURGUNDY,
    )

    plot_style(
        ax_fc,
        jd["steps"],
        jd["energy_fuelcell"],
        title="Fuel Cell Energy",
        ylabel="Energy [GJ/h]",
        line_color=BURGUNDY,
    )

    h2_storage_t = [float(v) / 120.0 for v in jd["hydrogen_storage"]]
    plot_style(
        ax_h2,
        jd["steps"],
        h2_storage_t,
        title="Hydrogen Storage",
        ylabel="Hydrogen Storage [t]",
        line_color=BURGUNDY,
    )

    ax_h2.set_xlabel("Time [h]", fontsize=9)
    for ax in (ax_trains, ax_turb, ax_fc):
        ax.tick_params(labelbottom=False)
    _set_max_ticks(ax_h2, x_steps, max_ticks)

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.subplots_adjust(hspace=0.25)
    fig.canvas.draw()
    return fig, axs
