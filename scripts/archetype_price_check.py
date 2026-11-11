"""Diagnostic (Task 1): confirm the ContractEnv spot-price BASIS paths differ
between seeds and match the NBP price_dynamics calibration in
src/hydro_bo/data/logistics/config.yml.

Runs one archetype point (low-NH3 reference design) for seeds 1000 and 1001,
pulls each env's OWN price-dynamics object (so the check is faithful to whatever
class wires the blue block), confirms the calibration, and advances the basis 10
days to print the first 10 daily values per seed.

Run via scripts/pbs/archetype_price_check.pbs — a PBS job. Do NOT run on a login node.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from hydro_bo.archetypes.pipeline import load_config, problems  # noqa: E402
from hydro_bo.archetypes.rollout import build_env  # noqa: E402
from hydro_bo.archetypes.scenarios import design_from_x  # noqa: E402

SEEDS = [1000, 1001]
N_DAYS = 10


def find_price_dynamics(env):
    """The price-dynamics object the env built, via common attribute names."""
    dyn = getattr(env, "_dynamics", None)
    for obj in (dyn, env):
        for attr in ("_price_dynamics", "price_dynamics"):
            pd = getattr(obj, attr, None)
            if pd is not None:
                return pd
    return None


def basis_value(pd) -> float:
    """The OU log-basis y (blue/fundamental mode) or log(price) otherwise."""
    y = getattr(pd, "y", None)
    return float(y) if y is not None else float(np.log(pd.price))


def main():
    cfg = load_config()
    problem = problems(cfg)[0]                       # low-NH3
    x = problem.bounds().mean(axis=1)                # midpoint design (price is design-independent)
    design = design_from_x(x, problem)
    print(f"problem={problem.key}  design=bounds-midpoint  seeds={SEEDS}", flush=True)

    series = {}
    params_seen = None
    for seed in SEEDS:
        env = build_env(design, problem, seed)
        pd = find_price_dynamics(env)
        if pd is None:
            raise SystemExit("ERROR: no price-dynamics object on the env — prices are OFF for this run")
        params = {k: getattr(pd, k, None) for k in ("kappa", "sigma", "lambda_", "jump_mu", "jump_sigma")}
        if params_seen is None:
            params_seen = params
            print(f"price model: {type(pd).__name__}", flush=True)
            print(f"  calibration (seed-independent): {params}", flush=True)
            print(f"  fundamental/blue block: {getattr(pd, 'fundamental', None)}", flush=True)
        b0 = basis_value(pd)
        daily = []
        for _ in range(N_DAYS):
            pd.step(dt=1 / 365)
            daily.append(basis_value(pd))
        series[seed] = daily
        print(f"seed {seed}: y0={b0:+.6f}  first {N_DAYS} daily basis = "
              + ", ".join(f"{v:+.6f}" for v in daily), flush=True)

    a = np.array(series[SEEDS[0]])
    b = np.array(series[SEEDS[1]])
    print(f"\nmax |basis diff| between seed {SEEDS[0]} and {SEEDS[1]} = {np.max(np.abs(a - b)):.6f}", flush=True)
    print("RESULT: price paths DIFFER between seeds" if not np.allclose(a, b)
          else "RESULT: price paths IDENTICAL — seed not reaching the price stream!", flush=True)


if __name__ == "__main__":
    main()
