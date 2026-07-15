"""Wiring for the idc / parametric split.

Pins the invariants that keep the two flavours independent: IDC configs
load without a theta block, the parametric config resolves its shared
planning model, the joint box is [design | theta], and theta dims are
frozen (not optimised) during acquisition.

Run directly:  python tests/test_parametric_wiring.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import yaml

from hydro_bo.opt.parametric import (
    ThetaFrozenConstrainedNLP,
    ThetaFrozenMixedIntNLP,
)
from hydro_bo.utils.run_config import load_config, planning_model_path
from hydro_bo.utils.search_space import PARAM_KEYS, build_bounds, build_cat_vars
from hydro_bo.utils.theta import registry_from_names

ROOT = Path(__file__).resolve().parents[1]
IDC = ROOT / "scripts" / "idc"
PARAMETRIC = ROOT / "scripts" / "parametric"


def test_idc_config_has_no_theta():
    cfg = load_config(IDC / "config.yml")
    assert cfg.theta is None, "IDC config must stay theta-free"
    assert cfg.general.planning_dir is None
    # default planning path is unchanged by the planning_dir addition
    assert planning_model_path(IDC, "NH3") == IDC / "tmp" / "planning" / "NH3-Chile.yml"
    print("test_idc_config_has_no_theta PASSED")


def test_parametric_config_and_shared_planning():
    cfg = load_config(PARAMETRIC / "config.yml")
    assert cfg.theta is not None and cfg.theta.params
    pm = planning_model_path(PARAMETRIC, cfg.general.vector, cfg.general.planning_dir)
    # planning_dir must resolve out of parametric/ and into the idc solve
    assert pm.resolve() == (IDC / "tmp" / "planning" / f"{cfg.general.vector}-Chile.yml").resolve(), pm
    print("test_parametric_config_and_shared_planning PASSED")


def test_joint_bounds_layout():
    cfg = load_config(PARAMETRIC / "config.yml")
    pm = planning_model_path(PARAMETRIC, cfg.general.vector, cfg.general.planning_dir)
    if not pm.exists():
        print("test_joint_bounds_layout SKIPPED (no planning model present)")
        return
    ref = yaml.safe_load(pm.read_text())
    reg = registry_from_names(cfg.theta.params)
    bx = build_bounds(ref, cfg.general.bounds_expansion)
    bounds = np.vstack([bx, reg.bounds()])

    assert bx.shape[0] == len(PARAM_KEYS)
    assert bounds.shape[0] == len(PARAM_KEYS) + reg.dim
    # theta occupies the trailing block, design box untouched
    np.testing.assert_array_equal(bounds[: len(PARAM_KEYS)], bx)
    np.testing.assert_array_equal(bounds[len(PARAM_KEYS):], reg.bounds())
    # integer dims must not collide with the theta block
    assert all(i < len(PARAM_KEYS) for i, _ in build_cat_vars(bx))
    print("test_joint_bounds_layout PASSED")


def _layout_for(cls, **extra):
    cat_vars = [(1, [0.0, 0.5, 1.0]), (12, [0.0, 1.0])]
    s = cls(cat_vars=cat_vars, theta_dims=(13, 14), theta_unit=(0.25, 0.75), **extra)
    return s._layout()


def test_theta_is_frozen_not_optimised():
    for cls, extra in ((ThetaFrozenConstrainedNLP, {"l1_penalty": 1.0}),
                       (ThetaFrozenMixedIntNLP, {})):
        idx, combos = _layout_for(cls, **extra)
        assert idx == (1, 12, 13, 14), idx
        assert len(combos) == 3 * 2, f"expected one combo per integer level, got {len(combos)}"
        for c in combos:
            # theta pinned at its node in every combo, at the sorted positions
            assert c[2] == 0.25 and c[3] == 0.75, c
        # design integer levels still enumerated
        assert {(c[0], c[1]) for c in combos} == {
            (a, b) for a in (0.0, 0.5, 1.0) for b in (0.0, 1.0)
        }
    print("test_theta_is_frozen_not_optimised PASSED")


def test_theta_overlapping_integer_dim_rejected():
    try:
        ThetaFrozenConstrainedNLP(
            cat_vars=[(1, [0.0, 1.0])], l1_penalty=1.0,
            theta_dims=(1,), theta_unit=(0.5,),
        )._layout()
    except ValueError:
        print("test_theta_overlapping_integer_dim_rejected PASSED")
        return
    raise AssertionError("expected ValueError on theta/integer dim overlap")


if __name__ == "__main__":
    test_idc_config_has_no_theta()
    test_parametric_config_and_shared_planning()
    test_joint_bounds_layout()
    test_theta_is_frozen_not_optimised()
    test_theta_overlapping_integer_dim_rejected()
    print("ALL PASSED")
