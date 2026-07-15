"""Central registry for uncertain parameters (theta).

Each `ThetaParam` names one uncertain quantity, its physical bounds, and
ALL the places it lands (its sinks). `ThetaRegistry.apply` turns a theta
vector into the two delivery channels the eval pipeline already has:

  - `cost_overrides`: dotted-path patches applied to the h2_plan
    parameter tree inside `calculate_capex_opex`.
  - `env_overrides`: nested dict deep-merged into the env config by the
    dispatcher worker. MPC params ride this channel at
    `mpc.param_overrides.<name>`, consumed by `import_mpc_data` (fresh
    build) or `MPCController.apply_param_updates` (in-place).

Sinks are validated: an `MpcParam` on a structural parameter (see
`IMMUTABLE_MPC_PARAMS`) is rejected unless `rebuild_ok=True`, and
`validate_runtime` checks every path/name resolves against the real
parameter tree and MPC param set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from hydro_bo.mpc.immutable import IMMUTABLE_MPC_PARAMS


@dataclass(frozen=True)
class CostCoeff:
    """Dotted path into h2_plan `formulation_parameters`, e.g.
    "capital_costs.electrolysers.SOFC.1" (integers index lists)."""

    path: str


@dataclass(frozen=True)
class MpcParam:
    """Flat parameter name consumed by `import_mpc_data` and built as a
    (mutable) Pyomo Param on the MPC instance."""

    name: str
    rebuild_ok: bool = False


@dataclass(frozen=True)
class EnvConfig:
    """Dotted path into the env config, e.g.
    "weather_data.forecast_mean_override"."""

    path: str


Sink = CostCoeff | MpcParam | EnvConfig


@dataclass(frozen=True)
class ThetaParam:
    name: str
    bounds: tuple[float, float]
    sinks: tuple[Sink, ...]
    nominal: float | None = None
    warp: str = "uniform"

    def __post_init__(self):
        lo, hi = self.bounds
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            raise ValueError(f"{self.name}: bad bounds {self.bounds}")
        if not self.sinks:
            raise ValueError(f"{self.name}: at least one sink required")
        if self.nominal is not None and not (lo <= self.nominal <= hi):
            raise ValueError(f"{self.name}: nominal {self.nominal} outside bounds")
        for s in self.sinks:
            if isinstance(s, MpcParam) and s.name in IMMUTABLE_MPC_PARAMS and not s.rebuild_ok:
                raise ValueError(
                    f"{self.name}: MPC param {s.name!r} is structural; "
                    "set rebuild_ok=True to accept rebuild cost"
                )


@dataclass(frozen=True)
class ThetaBundle:
    cost_overrides: dict = field(default_factory=dict)
    env_overrides: dict = field(default_factory=dict)


def _path_parts(path: str) -> list:
    return [int(p) if p.lstrip("-").isdigit() else p for p in path.split(".")]


def get_path(tree, path: str):
    node = tree
    for part in _path_parts(path):
        node = node[part]
    return node


def set_path(tree, path: str, value) -> None:
    parts = _path_parts(path)
    node = tree
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]]  # raise KeyError/IndexError before writing
    node[parts[-1]] = value


def _set_nested(target: dict, keys: Sequence[str], value) -> None:
    d = target
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


class ThetaRegistry:
    def __init__(self, params: Sequence[ThetaParam]):
        names = [p.name for p in params]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate theta names: {names}")
        self.params = list(params)

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.params]

    @property
    def dim(self) -> int:
        return len(self.params)

    def bounds(self) -> np.ndarray:
        return np.array([list(p.bounds) for p in self.params], dtype=float)

    def nominal(self) -> np.ndarray:
        return np.array(
            [p.nominal if p.nominal is not None else 0.5 * sum(p.bounds)
             for p in self.params],
            dtype=float,
        )

    def apply(self, theta: np.ndarray) -> ThetaBundle:
        theta = np.asarray(theta, dtype=float).ravel()
        if theta.shape[0] != self.dim:
            raise ValueError(f"theta has dim {theta.shape[0]}, registry has {self.dim}")
        cost: dict = {}
        env: dict = {}
        for p, val in zip(self.params, theta):
            lo, hi = p.bounds
            if not (lo - 1e-12 <= val <= hi + 1e-12):
                raise ValueError(f"{p.name}={val} outside bounds {p.bounds}")
            v = float(val)
            for s in p.sinks:
                if isinstance(s, CostCoeff):
                    cost[s.path] = v
                elif isinstance(s, MpcParam):
                    _set_nested(env, ("mpc", "param_overrides", s.name), v)
                elif isinstance(s, EnvConfig):
                    _set_nested(env, tuple(s.path.split(".")), v)
        return ThetaBundle(cost_overrides=cost, env_overrides=env)

    def validate_runtime(
        self,
        parameter_tree: Mapping | None = None,
        mpc_param_names: set | None = None,
    ) -> None:
        """Check every sink resolves against the real structures. Pass
        `parameter_tree` (h2_plan formulation_parameters) to check
        CostCoeff paths, `mpc_param_names` (keys of import_mpc_data's
        params) to check MpcParam names."""
        errors = []
        for p in self.params:
            for s in p.sinks:
                if isinstance(s, CostCoeff) and parameter_tree is not None:
                    try:
                        node = get_path(parameter_tree, s.path)
                    except (KeyError, IndexError, TypeError):
                        errors.append(f"{p.name}: cost path {s.path!r} does not resolve")
                        continue
                    if not isinstance(node, (int, float)):
                        errors.append(f"{p.name}: cost path {s.path!r} is not a scalar")
                elif isinstance(s, MpcParam) and mpc_param_names is not None:
                    if s.name not in mpc_param_names:
                        errors.append(f"{p.name}: MPC param {s.name!r} not consumed by import_mpc_data")
        if errors:
            raise ValueError("theta registry validation failed:\n  " + "\n  ".join(errors))


def triple_bounds(parameter_tree: Mapping, path: str) -> tuple[float, float, float]:
    """(lo, nominal, hi) from an h2_plan data triple. Triples are stored
    [optimistic, nominal, pessimistic] and may run in either direction;
    the nominal is always the middle entry."""
    node = get_path(parameter_tree, path)
    if not (isinstance(node, (list, tuple)) and len(node) == 3):
        raise ValueError(f"{path!r} is not a data triple: {node!r}")
    a, mid, b = (float(v) for v in node)
    lo, hi = min(a, b), max(a, b)
    if not lo <= mid <= hi:
        raise ValueError(f"{path!r} nominal is outside its endpoints: {node!r}")
    return lo, mid, hi


def default_catalog() -> dict[str, ThetaParam]:
    """Verified uncertain parameters, bounds from h2_plan's own
    [lo, mid, hi] data triples. Select by name to build a registry."""
    from h2_plan.data import DefaultParams

    tree = DefaultParams("default").formulation_parameters

    def from_triple(name, triple_path, sinks):
        lo, mid, hi = triple_bounds(tree, triple_path)
        return ThetaParam(name=name, bounds=(lo, hi), nominal=mid, sinks=tuple(sinks))

    catalog = {}

    catalog["electrolyser_capex"] = from_triple(
        "electrolyser_capex", "capital_costs.electrolysers.SOFC",
        [CostCoeff("capital_costs.electrolysers.SOFC.1")],
    )
    catalog["turbine_capex"] = from_triple(
        "turbine_capex", "capital_costs.turbine",
        [CostCoeff("capital_costs.turbine.1")],
    )
    catalog["discount_factor"] = from_triple(
        "discount_factor", "miscillaneous.discount_factor",
        [CostCoeff("miscillaneous.discount_factor.1"), MpcParam("discount_factor")],
    )
    catalog["electrolysis_efficiency"] = from_triple(
        "electrolysis_efficiency", "efficiencies.electrolysers.SOFC",
        [MpcParam("electrolysis_efficiency")],
    )
    catalog["fuelcell_efficiency"] = from_triple(
        "fuelcell_efficiency", "efficiencies.fuel_cell",
        [MpcParam("fuelcell_efficiency")],
    )
    catalog["compression_efficiency"] = from_triple(
        "compression_efficiency", "efficiencies.compressor",
        [MpcParam("compression_efficiency")],
    )
    return catalog


def registry_from_names(names: Sequence[str]) -> ThetaRegistry:
    catalog = default_catalog()
    unknown = [n for n in names if n not in catalog]
    if unknown:
        raise KeyError(f"unknown theta params {unknown}; catalog has {sorted(catalog)}")
    return ThetaRegistry([catalog[n] for n in names])
