"""
Build and solve the logistics model once for the stage-0 designs, and report the
magnitude of every variable and every constraint coefficient.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml
from pyomo.environ import Constraint, Objective, Var, value
from pyomo.repn import generate_standard_repn

from hydro_bo.envs.shipping.utils import import_mpc_data, import_mpc_functions
from hydro_bo.mpc.mpc import LogisticsMPCController
from hydro_bo.utils.search_space import split_env_overrides


ROOT_DIR = Path(__file__).resolve().parents[1]
LOGISTICS_DATA_DIR = ROOT_DIR / "src" / "hydro_bo" / "data" / "logistics"
LOGISTICS_EQUATIONS_PATH = ROOT_DIR / "src" / "hydro_bo" / "envs" / "contract" / "logistics_equations.py"
STAGE4_DIR = ROOT_DIR / "paper_results" / "contract" / "stage4"

DESIGNS = {
    "NH3": ROOT_DIR / "paper_results" / "ablation" / "NH3" / "sobol_64_CI_NONE_comb_08"
    / "NH3-sobol_64_CI_NONE_comb_08-best.yml",
    "LH2": ROOT_DIR / "paper_results" / "ablation" / "LH2" / "sobol_64_CI_050_comb_14"
    / "LH2-sobol_64_CI_050_comb_14-best.yml",
}

PRICE = 5.0           # $/kg
MONTH = 1             # first env month of the stage-4 site rollouts
ZERO_TOL = 1e-6       # |x| below this counts as zero
SCALE_LOWER = 1e-3
SCALE_UPPER = 1e3


def month_mean_delivered(label):
    """Mean daily delivered vector [t/day] over MONTH of the stage-4 site rollouts, all seeds."""
    daily = []
    for path in sorted(STAGE4_DIR.glob(f"{label}-site-seed_*.npz")):
        d = np.load(path)
        daily.append(d["daily_delivered_t"][d["day_month"] == MONTH])
    return float(np.concatenate(daily).mean())


def build_logistics(design, delivered):
    params, _ = split_env_overrides(design)
    controller_data = import_mpc_data(
        dict(params),
        design["vector"],
        data_folder=LOGISTICS_DATA_DIR,
    )
    controller_data.update(
        import_mpc_functions(
            LOGISTICS_DATA_DIR,
            controller_data["sets"],
            equations_path=LOGISTICS_EQUATIONS_PATH,
        )
    )
    controller = LogisticsMPCController(gurobi_seed=0)
    controller.build(controller_data)

    for t in controller_data["sets"]["grid0"]:
        controller.model.vector_delivered[t] = delivered
        controller.model.spot_forecast[t] = PRICE
        controller.model.fundamental_price[t] = PRICE
    return controller


def magnitudes(values):
    """Minimum non-zero and maximum absolute value."""
    values = np.abs(np.asarray(values, dtype=float))
    nonzero = values[values > ZERO_TOL]
    return (nonzero.min() if nonzero.size else np.nan), (values.max() if values.size else np.nan)


def flag(lower, upper):
    flags = []
    if lower < SCALE_LOWER:
        flags.append(f"min < {SCALE_LOWER:g}")
    if upper > SCALE_UPPER:
        flags.append(f"max > {SCALE_UPPER:g}")
    return ", ".join(flags)


def variable_rows(inst):
    rows = []
    for var in inst.component_objects(Var, active=True):
        values = [value(var[i], exception=False) or 0.0 for i in var]
        rows.append((var.name, len(values), *magnitudes(values)))
    return rows


def constraint_rows(inst):
    rows = []
    for con in inst.component_objects(Constraint, active=True):
        coefs, constants = [], []
        for i in con:
            repn = generate_standard_repn(con[i].body)
            coefs += [float(c) for c in repn.linear_coefs]
            body_constant = float(value(repn.constant))
            for bound in (con[i].lower, con[i].upper):
                if bound is not None:
                    constants.append(float(value(bound)) - body_constant)
        rows.append((con.name, len(con), *magnitudes(coefs), *magnitudes(constants)))

    for obj in inst.component_objects(Objective, active=True):
        repn = generate_standard_repn(obj.expr)
        rows.append((f"{obj.name} (objective)", 1, *magnitudes(repn.linear_coefs),
                     *magnitudes([value(repn.constant)])))
    return rows


def report(vector, controller, delivered):
    inst = controller.instance
    print(f"\n=== {vector}  (vector_delivered {delivered:.1f} t/day, price {PRICE:g} $/kg) ===")
    print(f"objective = {value(inst.obj1):.3f} $M")

    print("\nVariables")
    print(f"  {'name':<26} {'n':>5}  {'min |x|>0':>11}  {'max |x|':>11}  flag")
    for name, n, lower, upper in variable_rows(inst):
        print(f"  {name:<26} {n:>5}  {lower:>11.3e}  {upper:>11.3e}  {flag(lower, upper)}")

    print("\nConstraint coefficients (and right-hand sides)")
    print(f"  {'block':<34} {'n':>4}  {'min |a|>0':>11}  {'max |a|':>11}  "
          f"{'min |b|>0':>11}  {'max |b|':>11}  flag")
    for name, n, lower, upper, b_lower, b_upper in constraint_rows(inst):
        print(f"  {name:<34} {n:>4}  {lower:>11.3e}  {upper:>11.3e}  "
              f"{b_lower:>11.3e}  {b_upper:>11.3e}  {flag(lower, upper)}")


def main():
    for vector, design_path in DESIGNS.items():
        with open(design_path) as f:
            design = yaml.safe_load(f)
        delivered = month_mean_delivered(design_path.stem.removesuffix("-best"))
        controller = build_logistics(design, delivered)
        controller.solve(solver="gurobi")
        report(vector, controller, delivered)


if __name__ == "__main__":
    main()
