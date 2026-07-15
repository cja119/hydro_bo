"""Update-vs-rebuild parity for MPC parameters, at the Gurobi matrix level.

For each parameter a theta sink can target, an in-place
`apply_param_updates` must yield the SAME Gurobi model (objective
coefficients, constraint rows, RHS, bounds) as a fresh build at that
value — this is the guarantee that lets theta sweeps skip rebuilds. It
also asserts the perturbation changes the matrix at all: a parameter
that alters nothing is not actually consumed, i.e. a broken sink.

Needs a Gurobi licence. Run directly:  python tests/test_mpc_update_parity.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pyomo.environ import Objective, value

from hydro_bo.envs.shipping.utils import import_mpc_data, import_mpc_functions
from hydro_bo.mpc.mpc import MPCController

PLANNING = {
    "compression_capacity": 100.0, "conversion_trains_number": 2,
    "electrolyser_capacity": 500.0, "fuelcell_capacity": 50.0,
    "hydrogen_storage_capacity": 1000.0, "renewable_energy_capacity": 800.0,
    "vector_storage_capacity": 2000.0, "capex": 1.0, "opex": 1.0,
    "renewables": "wind", "expected_arrival_offset": 0,
}

TARGETS = {
    "electrolysis_efficiency": 0.85,      # coefficient position
    "fuelcell_efficiency": 0.85,
    "compression_efficiency": 0.9,
    "renewable_energy_capacity": 0.7,     # RHS position
    "hydrogen_storage_lower_backoff": 1.5,
    "discount_factor": 1.5,               # objective side
}

TOL = 1e-9


def _build_controller(param_overrides=None):
    data = import_mpc_data(dict(PLANNING), "NH3", param_overrides=param_overrides)
    data.update(import_mpc_functions(None, data["sets"]))
    c = MPCController(gurobi_seed=0)
    c.build(data)
    c._ensure_solver("gurobi")
    c.instance = c.model.create_instance()
    c.solver.set_instance(c.instance)
    c._instance_bound = True
    c._snapshot_relaxation_baselines()
    grid1 = data["sets"]["grid1"]
    c.apply_param_updates({
        "energy_wind": {t: 6.0 for t in data["sets"]["grid0"]},
        "h2_price": 5.0,
        "ship_arrived": {"small": 0, "medium": 1, "large": 0},
        "expected_ships": {
            (s, t): (1 if (s == "medium" and t == grid1[len(grid1) // 2]) else 0)
            for s in data["sets"]["ships"]
            for t in grid1
        },
    })
    return c


def _fingerprint(c):
    m = c.solver._solver_model
    m.update()
    fp = {
        "dims": (m.NumVars, m.NumConstrs),
        "obj": {v.VarName: v.Obj for v in m.getVars()},
        "bounds": {v.VarName: (v.LB, v.UB, v.VType) for v in m.getVars()},
    }
    rows = {}
    for con in m.getConstrs():
        row = m.getRow(con)
        terms = tuple(sorted(
            (row.getVar(i).VarName, row.getCoeff(i)) for i in range(row.size())
        ))
        rows[con.ConstrName] = (con.Sense, con.RHS, terms)
    fp["rows"] = rows
    return fp


def _diff(fp_a, fp_b):
    """List of human-readable differences between two fingerprints."""
    out = []
    if fp_a["dims"] != fp_b["dims"]:
        out.append(f"dims {fp_a['dims']} != {fp_b['dims']}")
        return out
    for key in ("obj", "bounds"):
        for name, va in fp_a[key].items():
            vb = fp_b[key].get(name)
            if vb is None:
                out.append(f"{key}: {name} missing")
            elif isinstance(va, tuple):
                if va[2] != vb[2] or any(abs(x - y) > TOL for x, y in zip(va[:2], vb[:2])):
                    out.append(f"{key}: {name} {va} != {vb}")
            elif abs(va - vb) > TOL:
                out.append(f"{key}: {name} {va} != {vb}")
    for name, (sense_a, rhs_a, terms_a) in fp_a["rows"].items():
        rb = fp_b["rows"].get(name)
        if rb is None:
            out.append(f"row missing: {name}")
            continue
        sense_b, rhs_b, terms_b = rb
        if sense_a != sense_b or abs(rhs_a - rhs_b) > TOL:
            out.append(f"row {name}: sense/rhs differ")
        elif len(terms_a) != len(terms_b) or any(
            na != nb or abs(ca - cb) > TOL
            for (na, ca), (nb, cb) in zip(terms_a, terms_b)
        ):
            out.append(f"row {name}: coefficients differ")
    return out


def main():
    base = _build_controller()
    base_fp = _fingerprint(base)
    base_vals = {k: float(value(getattr(base.instance, k))) for k in TARGETS}

    base.solve()
    objs = list(base.instance.component_objects(Objective, active=True))
    print(f"nominal model solves; objective = {float(value(objs[0])):.6f}")

    failures = []
    for name, factor in TARGETS.items():
        v_new = base_vals[name] * factor

        fresh = _build_controller(param_overrides={name: v_new})
        fp_rebuild = _fingerprint(fresh)

        base.apply_param_updates({name: v_new})
        fp_update = _fingerprint(base)
        base.apply_param_updates({name: base_vals[name]})

        mismatches = _diff(fp_rebuild, fp_update)
        touched = len(_diff(base_fp, fp_rebuild))
        status = "OK" if not mismatches and touched else (
            "MISMATCH" if mismatches else "NOT CONSUMED"
        )
        print(f"{name}: matrix_entries_changed={touched} "
              f"update_vs_rebuild_diffs={len(mismatches)} [{status}]")
        for m in mismatches[:5]:
            print(f"    {m}")
        if mismatches:
            failures.append((name, "update != rebuild"))
        if not touched:
            failures.append((name, "perturbation changed nothing — sink not consumed"))

    if failures:
        raise AssertionError(f"parity failures: {failures}")
    print("ALL PASSED")


if __name__ == "__main__":
    main()
