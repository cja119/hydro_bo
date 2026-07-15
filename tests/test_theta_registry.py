"""Theta registry: sink routing, validation, and the cost/MPC hooks.

Run directly (no pytest needed):  python tests/test_theta_registry.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from hydro_bo.utils.theta import (
    CostCoeff,
    EnvConfig,
    MpcParam,
    ThetaParam,
    ThetaRegistry,
    default_catalog,
    registry_from_names,
    set_path,
    triple_bounds,
)


def test_sink_routing():
    reg = ThetaRegistry([
        ThetaParam("a", (0.0, 1.0), (CostCoeff("capital_costs.turbine.1"),)),
        ThetaParam("b", (0.5, 0.9), (MpcParam("electrolysis_efficiency"),)),
        ThetaParam("c", (0.0, 1.0), (EnvConfig("weather_data.forecast_mean_override"),)),
        ThetaParam("d", (0.0, 2.0), (CostCoeff("x.y"), MpcParam("discount_factor"))),
    ])
    bundle = reg.apply(np.array([0.3, 0.7, 0.2, 1.5]))
    assert bundle.cost_overrides == {"capital_costs.turbine.1": 0.3, "x.y": 1.5}
    assert bundle.env_overrides["mpc"]["param_overrides"] == {
        "electrolysis_efficiency": 0.7, "discount_factor": 1.5,
    }
    assert bundle.env_overrides["weather_data"]["forecast_mean_override"] == 0.2
    assert reg.dim == 4 and reg.bounds().shape == (4, 2)
    print("test_sink_routing PASSED")


def test_construction_validation():
    for bad in [
        lambda: ThetaParam("x", (1.0, 0.0), (CostCoeff("a"),)),         # inverted bounds
        lambda: ThetaParam("x", (0.0, 1.0), ()),                        # no sinks
        lambda: ThetaParam("x", (0.0, 1.0), (MpcParam("forecast_horizon"),)),  # structural
        lambda: ThetaRegistry([
            ThetaParam("x", (0.0, 1.0), (CostCoeff("a"),)),
            ThetaParam("x", (0.0, 1.0), (CostCoeff("b"),)),
        ]),                                                             # duplicate name
    ]:
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")

    ThetaParam("ok", (0.0, 1.0), (MpcParam("forecast_horizon", rebuild_ok=True),))

    reg = ThetaRegistry([ThetaParam("a", (0.0, 1.0), (CostCoeff("p"),))])
    try:
        reg.apply(np.array([2.0]))
        raise AssertionError("expected out-of-bounds ValueError")
    except ValueError:
        pass
    print("test_construction_validation PASSED")


def test_set_path_strictness():
    tree = {"a": {"b": [1.0, 2.0, 3.0]}}
    set_path(tree, "a.b.1", 9.0)
    assert tree["a"]["b"][1] == 9.0
    try:
        set_path(tree, "a.zzz", 1.0)
        raise AssertionError("expected KeyError")
    except KeyError:
        pass
    assert triple_bounds({"t": [1.0, 2.0, 3.0]}, "t") == (1.0, 2.0, 3.0)
    print("test_set_path_strictness PASSED")


def test_cost_override_hits_capex():
    from hydro_bo.envs.shipping.utils import calculate_capex_opex

    kw = dict(
        renewables="wind", vector="NH3",
        compression_capacity=100.0, electrolyser_capacity=500.0,
        fuelcell_capacity=50.0, conversion_trains_number=2,
        hydrogen_storage_capacity=1000.0, renewable_energy_capacity=800.0,
        vector_storage_capacity=2000.0,
    )
    base = calculate_capex_opex(**kw)
    bumped = calculate_capex_opex(
        **kw, cost_overrides={"capital_costs.electrolysers.SOFC.1": 1e9},
    )
    assert bumped["capex"] > base["capex"] * 10
    again = calculate_capex_opex(**kw)
    assert abs(again["capex"] - base["capex"]) < 1e-9  # no leaked mutation

    try:
        calculate_capex_opex(**kw, cost_overrides={"capital_costs.not_a_thing": 1.0})
        raise AssertionError("expected KeyError")
    except KeyError:
        pass
    print("test_cost_override_hits_capex PASSED")


def _planning_dict():
    return {
        "compression_capacity": 100.0, "conversion_trains_number": 2,
        "electrolyser_capacity": 500.0, "fuelcell_capacity": 50.0,
        "hydrogen_storage_capacity": 1000.0, "renewable_energy_capacity": 800.0,
        "vector_storage_capacity": 2000.0, "capex": 1.0, "opex": 1.0,
        "renewables": "wind", "expected_arrival_offset": 0,
    }


def test_mpc_param_overrides():
    from hydro_bo.envs.shipping.utils import import_mpc_data

    data = import_mpc_data(_planning_dict(), "NH3")
    base_eff = data["params"]["electrolysis_efficiency"]

    data2 = import_mpc_data(
        _planning_dict(), "NH3",
        param_overrides={"electrolysis_efficiency": base_eff * 0.9},
    )
    assert abs(data2["params"]["electrolysis_efficiency"] - base_eff * 0.9) < 1e-12

    try:
        import_mpc_data(_planning_dict(), "NH3", param_overrides={"not_a_param": 1.0})
        raise AssertionError("expected KeyError")
    except KeyError:
        pass
    print("test_mpc_param_overrides PASSED")


def test_catalog_validates_against_runtime():
    from h2_plan.data import DefaultParams
    from hydro_bo.envs.shipping.utils import import_mpc_data

    catalog = default_catalog()
    reg = ThetaRegistry(list(catalog.values()))
    tree = DefaultParams("default").formulation_parameters
    data = import_mpc_data(_planning_dict(), "NH3")
    reg.validate_runtime(
        parameter_tree=tree, mpc_param_names=set(data["params"]),
    )
    for p in reg.params:
        lo, hi = p.bounds
        assert lo < hi and p.nominal is not None and lo <= p.nominal <= hi

    reg2 = registry_from_names(["electrolysis_efficiency", "discount_factor"])
    assert reg2.dim == 2
    print("test_catalog_validates_against_runtime PASSED")


if __name__ == "__main__":
    test_sink_routing()
    test_construction_validation()
    test_set_path_strictness()
    test_cost_override_hits_capex()
    test_mpc_param_overrides()
    test_catalog_validates_against_runtime()
    print("ALL PASSED")
