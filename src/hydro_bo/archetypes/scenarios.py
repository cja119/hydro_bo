"""Adoption-archetype scenarios for the contract-plus-design Bayesian optimisation.

A *problem* is one (archetype, carrier) pair. Each archetype fixes the export scale (conversion throughput, H2 LHV), the
green premium and the spot-market liquidity (half-spread and depth); the design variables are the plant capacities and
the contract factor. Conversion trains are fixed per problem: the fewest full-size trains that reach the target, at
least two, each resized to hit the target exactly.

Search bounds are +/- `bounds_expansion` around a reference design: the existing BO design for the carrier, with every
capacity scaled by the ratio of conversion throughputs. Port storage is floored at a multiple of one small-ship cargo so
the closed loop can always fill a ship.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]

# Full-size conversion train [t carrier/h] and carrier calorific value [GJ/t] (h2_plan defaults)
FULL_TRAIN = {"NH3": 114.0, "LH2": 6.25}
CALORIFIC = {"NH3": 18.8, "LH2": 120.0}
SMALL_CARGO_KT = {"NH3": 32.5, "LH2": 3.5425}          # small-ship capacity [kt carrier]
MIN_TRAINS = 2

# Design variables searched by the BO, in order. All continuous.
CAPACITY_KEYS = [
    "renewable_energy_capacity",
    "electrolyser_capacity",
    "hydrogen_storage_capacity",
    "compression_capacity",
    "fuelcell_capacity",
    "vector_storage_capacity",
]
SEARCH_KEYS = CAPACITY_KEYS + ["contract_factor"]


def train_mw(vector: str, tonnes_per_hour: float) -> float:
    """Conversion throughput of a train [MW, H2 LHV] from its rate [t carrier/h]."""
    return tonnes_per_hour * CALORIFIC[vector] / 3.6


def trains_for(vector: str, target_mw: float) -> tuple[int, float]:
    """(number of trains, train size [t carrier/h]) for a target export throughput [MW, H2 LHV]."""
    full_mw = train_mw(vector, FULL_TRAIN[vector])
    n = max(MIN_TRAINS, math.ceil(target_mw / full_mw - 1e-9))
    return n, target_mw / n * 3.6 / CALORIFIC[vector]


@dataclass(frozen=True)
class Archetype:
    name: str
    target_mw: float          # export conversion throughput [MW, H2 LHV]
    premium: float            # green premium nu [USD/kg]
    half_spread: float        # spot half-spread eps [-]
    depth_days: float         # market depth q_ref in days of the hub's own reference output [-]
    penalty: float = 9.0      # cap on the premium in the spot price [USD/kg]


@dataclass
class Problem:
    archetype: Archetype
    vector: str
    reference: dict
    bounds_expansion: float = 0.5
    contract_bounds: tuple = (0.5, 1.0)
    utilisation: float = 0.6            # reference utilisation of conversion nameplate for the contract volume
    min_port_cargoes: float = 1.2       # port storage floor, in small-ship cargoes
    trains: int = field(init=False)
    train_size: float = field(init=False)

    def __post_init__(self):
        self.trains, self.train_size = trains_for(self.vector, self.archetype.target_mw)

    @property
    def key(self) -> str:
        return f"{self.archetype.name}-{self.vector}"

    @property
    def scale(self) -> float:
        """Conversion throughput of this problem relative to the reference design."""
        ref_n = int(self.reference["conversion_trains_number"])
        return (self.trains * self.train_size) / (ref_n * FULL_TRAIN[self.vector])

    def reference_design(self) -> dict:
        return {k: float(self.reference[k]) * self.scale for k in CAPACITY_KEYS}

    def bounds(self) -> np.ndarray:
        ref = self.reference_design()
        rows = []
        for k in CAPACITY_KEYS:
            lo, hi = ref[k] * (1 - self.bounds_expansion), ref[k] * (1 + self.bounds_expansion)
            if k == "vector_storage_capacity":
                floor = self.min_port_cargoes * SMALL_CARGO_KT[self.vector]
                lo, hi = max(lo, floor), max(hi, 2 * floor)
            rows.append([lo, hi])
        rows.append(list(self.contract_bounds))
        return np.array(rows, dtype=float)

    # --- contract and market quantities -------------------------------------------------------------------------
    def reference_output_kt30(self, large_fuel_share: float) -> float:
        """Reference landed output X1 [kt H2 per 30 days]: nameplate conversion x utilisation, less voyage fuel."""
        h2_t_per_h = self.trains * self.train_size * CALORIFIC[self.vector] / 120
        return h2_t_per_h * 24 * 30 / 1000 * self.utilisation * (1 - large_fuel_share)

    def reference_delivery_t_per_day(self) -> float:
        """Reference site delivery [t carrier/day] used as the flat monthly delivery forecast beyond the site plan."""
        return self.trains * self.train_size * 24 * self.utilisation


def default_archetypes() -> dict[str, Archetype]:
    return {
        "low": Archetype("low", target_mw=500.0, premium=7.0, half_spread=0.2, depth_days=0.25),
        "medium": Archetype("medium", target_mw=1000.0, premium=5.0, half_spread=0.05, depth_days=1.0),
        "high": Archetype("high", target_mw=2000.0, premium=3.0, half_spread=0.002, depth_days=4.0),
    }


def load_problems(config: dict) -> list[Problem]:
    """Problems from the `archetype_bo` config block (archetype overrides, reference designs, bounds)."""
    archetypes = default_archetypes()
    for name, over in (config.get("archetypes") or {}).items():
        base = archetypes[name].__dict__ if name in archetypes else {"name": name}
        archetypes[name] = Archetype(**{**base, **over, "name": name})
    problems = []
    for name in config.get("problem_archetypes", list(archetypes)):
        for vector in config.get("vectors", ["NH3", "LH2"]):
            with open(ROOT / config["reference_designs"][vector]) as f:
                reference = yaml.safe_load(f)
            problems.append(Problem(archetypes[name], vector, reference,
                                    bounds_expansion=float(config.get("bounds_expansion", 0.5)),
                                    contract_bounds=tuple(config.get("contract_bounds", (0.5, 1.0))),
                                    utilisation=float(config.get("utilisation", 0.6)),
                                    min_port_cargoes=float(config.get("min_port_cargoes", 1.2))))
    return problems


def design_from_x(x: np.ndarray, problem: Problem, renewables: str = "wind") -> dict:
    """Design dict (planning-model keys, fixed trains and train size, contract factor, capex/opex) from a BO vector."""
    from hydro_bo.envs.shipping.utils import calculate_capex_opex

    x = np.asarray(x, dtype=float)
    design = {k: float(v) for k, v in zip(SEARCH_KEYS, x)}
    design["conversion_trains_number"] = problem.trains
    design["single_train_limit_conversion"] = problem.train_size
    design["renewables"] = renewables
    design["vector"] = problem.vector
    costs = calculate_capex_opex(
        renewables=renewables, vector=problem.vector,
        compression_capacity=design["compression_capacity"], electrolyser_capacity=design["electrolyser_capacity"],
        fuelcell_capacity=design["fuelcell_capacity"], conversion_trains_number=problem.trains,
        hydrogen_storage_capacity=design["hydrogen_storage_capacity"],
        renewable_energy_capacity=design["renewable_energy_capacity"],
        vector_storage_capacity=design["vector_storage_capacity"], single_train_throughput=problem.train_size,
    )
    design["capex"], design["opex"] = costs["capex"], costs["opex"]
    return design
