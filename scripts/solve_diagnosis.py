"""
Diagnose the slowest logistics solves of the all-ships 12g runs: rebuild each solve
from the saved daily arrays, save it as an LP file with the previous day's solution,
and solve it with Gurobi under several settings and formulation variants.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json
import re

import gurobipy as gp
import numpy as np
import yaml
from gurobipy import GRB

import logistics_rollout as lr
import logistics_sweep as ls
from hydro_bo.utils.logging_config import configure_logging, get_logger
from hydro_bo.utils.seeding import derive_subseed, gurobi_seed

logger = get_logger(__name__)

CONTRACT_DIR = Path(__file__).resolve().parents[1] / "paper_results" / "contract"
RUN_DIR = CONTRACT_DIR / "stage5g"
OUT_DIR = CONTRACT_DIR / "stage5h"

EPSILONS = [0.05, 0.2]
N_SLOWEST = 10                  # per carrier, across both half-spreads
MIP_GAP = 0.005
TIME_LIMIT = 120                # s, variants
PREVIOUS_TIME_LIMIT = 60        # s, previous day, as in the 12g runs
EARLY_TIME = 10                 # s
CUTOFF_DAYS = [14, 21, 28]
SHIP_VARS = ("n_ship_ordered", "n_ship_sent", "n_ship_discharged")
DAY_ONE = 24                    # t of the day carried out
ROLLOUT_OPTIONS = {             # as MPCController._ensure_solver
    "FeasibilityTol": 1e-6,
    "OptimalityTol": 1e-9,
    "Presolve": 2,
    "Cuts": -1,
    "Heuristics": 0.1,
    "Threads": 1,
    "Method": 1,
    "Seed": gurobi_seed(derive_subseed(lr.SEED, "gurobi")),
}
NAME = re.compile(r"^(\w+?)\((?:(\w+?)_)?(\d+)\)$")


def run_path(vector, eps):
    return RUN_DIR / f"run_{vector}_eps{eps:g}.npz"


def instance_name(vector, eps, day):
    return f"{vector}-eps{eps:g}-day{day}"


def select_instances(vectors, epsilons=EPSILONS):
    """The N_SLOWEST solves per carrier (day >= 2), as (vector, eps, day, solve_s)."""
    instances = []
    for vector in vectors:
        rows = []
        for eps in epsilons:
            d = np.load(run_path(vector, eps))
            rows += [(vector, eps, day, float(d["solve_s"][day])) for day in range(2, len(d["day"]))]
        instances += sorted(rows, key=lambda r: -r[3])[:N_SLOWEST]
    return instances


# ---------------------------------------------------------------------------
# Rebuild and save
# ---------------------------------------------------------------------------

def start_values(d, day):
    """End states of the solve on `day`, as the next solve fixes them at t=0."""
    values = {
        ("vector_storage", 0): float(d["storage_end"][day]),
        ("vector_flared", 0): float(d["flared"][day]),
        ("buyer_inventory", 0): float(d["buyer_inventory"][day]),
        ("spot_bought", 0): float(d["spot_bought"][day]),
        ("spot_sold", 0): float(d["spot_sold"][day]),
    }
    for j, size in enumerate(str(s) for s in d["ships"]):
        values["waiting_ships", (size, 0)] = float(d["waiting_model"][day, j])
        values["ships_holding", (size, 0)] = float(d["holding_model"][day, j])
        values["n_ship_ordered", (size, 0)] = int(d["ordered"][day, j])
        values["n_ship_sent", (size, 0)] = int(d["sent"][day, j])
        values["n_ship_discharged", (size, 0)] = int(d["discharged"][day, j])
    return values


def set_dynamics_day(dynamics, d, day):
    """Ship records as they stood at the start of `day`."""
    dynamics.iter_count = day
    dynamics.ships = []
    records = zip(d["ship_size"], d["ship_ordered"], d["ship_origin_arrival"], d["ship_sent"],
                  d["ship_buyer_arrival"], d["ship_discharged"])
    for size, ordered, origin, sent, buyer, discharged in records:
        if ordered >= day:
            continue
        arrived = 0 <= origin < day
        was_sent = 0 <= sent < day
        dynamics.ships.append({
            "size": str(size),
            "ordered": int(ordered),
            "eta": float(ordered + dynamics._arrival_time - day),
            "origin_arrival": int(origin) if arrived else None,
            "sent": int(sent) if was_sent else None,
            "buyer_arrival": int(buyer) if was_sent else None,
            "discharged": int(discharged) if 0 <= discharged < day else None,
        })


def export_instance(vector, eps, day):
    """Write the LP files of `day` and `day - 1`, rebuilt from the saved run."""
    d = np.load(run_path(vector, eps))
    with open(lr.DESIGNS[vector]) as f:
        design = yaml.safe_load(f)
    delivered = np.load(lr.STAGE4_DIR / f"{ls.label_of(vector)}-site-seed_{lr.SEED}.npz")["daily_delivered_t"]
    _, _, _, settings = next(g for g in ls.grid(vector, 0) if g[:3] == (0.0, 2.0, eps))
    controller, dynamics, _ = lr.build_logistics(design, delivered, settings)

    # First solve, to create the instance in its warm-start form
    controller.update(dynamics.get_mpc_args(), None)
    controller.solve(solver="gurobi")

    paths = {}
    for target in (day - 1, day):
        set_dynamics_day(dynamics, d, target)
        controller.update(dynamics.get_mpc_args(), start_values(d, target - 1))
        paths[target] = OUT_DIR / f"{instance_name(vector, eps, target)}.lp"
        controller.instance.write(str(paths[target]), io_options={"symbolic_solver_labels": True})
    return paths


# ---------------------------------------------------------------------------
# Solve variants
# ---------------------------------------------------------------------------

def load(lp_path, time_limit=TIME_LIMIT, **params):
    model = gp.read(str(lp_path))
    model.Params.OutputFlag = 0
    for key, val in {**ROLLOUT_OPTIONS, "MIPGap": MIP_GAP, "TimeLimit": time_limit, **params}.items():
        model.setParam(key, val)
    return model


def parse(name):
    """(base, size or None, t) from an LP variable name."""
    match = NAME.match(name)
    return (match.group(1), match.group(2), int(match.group(3))) if match else (None, None, None)


def set_shifted_start(model, previous):
    """Start each variable at t from the previous day's value at t + 24."""
    n_set = 0
    for var in model.getVars():
        base, size, t = parse(var.VarName)
        if base is None:
            continue
        key = f"{base}({size}_{t + 24})" if size else f"{base}({t + 24})"
        if key in previous:
            var.Start = previous[key]
            n_set += 1
    return n_set


def relax_after(model, cutoff_day):
    """Ship variables integral only up to `cutoff_day`."""
    for var in model.getVars():
        base, _, t = parse(var.VarName)
        if base in SHIP_VARS and t > cutoff_day * 24:
            var.VType = GRB.CONTINUOUS


def root_gap(model):
    """Relative gap after the root node, with branching switched off."""
    model.Params.NodeLimit = 1
    model.optimize()
    if not model.SolCount:
        return None
    return abs(model.ObjBound - model.ObjVal) / abs(model.ObjVal)


def build_variant(lp_path, params, warm, cutoff, previous):
    model = load(lp_path, **params)
    if cutoff is not None:
        relax_after(model, cutoff)
    if warm:
        set_shifted_start(model, previous)
    return model


def solve(model):
    log = {"gap_time": None, "incumbents": []}

    def callback(m, where):
        if where == GRB.Callback.MIP:
            best, bound = m.cbGet(GRB.Callback.MIP_OBJBST), m.cbGet(GRB.Callback.MIP_OBJBND)
            if log["gap_time"] is None and abs(best) < GRB.INFINITY and abs(bound - best) <= MIP_GAP * abs(best):
                log["gap_time"] = m.cbGet(GRB.Callback.RUNTIME)
        elif where == GRB.Callback.MIPSOL:
            log["incumbents"].append((m.cbGet(GRB.Callback.RUNTIME), m.cbGet(GRB.Callback.MIPSOL_OBJ)))

    model.optimize(callback)

    best_early = max((obj for t, obj in log["incumbents"] if t <= EARLY_TIME), default=None)
    solved = model.Status == GRB.OPTIMAL
    decisions = {}
    if model.SolCount:
        for var in model.getVars():
            base, size, t = parse(var.VarName)
            if t == DAY_ONE and (base in SHIP_VARS or base in ("spot_sold", "spot_bought")):
                decisions[f"{base}[{size}]" if size else base] = round(var.X, 4)
    return {
        "status": model.Status,
        "runtime": model.Runtime,
        "gap_time": log["gap_time"] if log["gap_time"] is not None else (model.Runtime if solved else None),
        "final_gap": model.MIPGap if model.SolCount else None,
        "best_early": best_early,
        "best_final": model.ObjVal if model.SolCount else None,
        "nodes": model.NodeCount,
        "decisions": decisions,
    }


def variants():
    """(label, Gurobi parameters, warm start, integrality cut-off day)."""
    out = [("E0", {}, False, None), ("E1", {"MIPFocus": 2, "Symmetry": 2, "Cuts": 2}, False, None),
           ("E2", {}, True, None)]
    out += [(f"E3-{c}", {}, False, c) for c in CUTOFF_DAYS]
    out += [("E4", {}, True, 21)]
    return out


def run_instance(vector, eps, day):
    name = instance_name(vector, eps, day)
    paths = export_instance(vector, eps, day)

    previous_model = load(paths[day - 1], time_limit=PREVIOUS_TIME_LIMIT)
    previous_model.optimize()
    previous_model.write(str(OUT_DIR / f"{instance_name(vector, eps, day - 1)}.sol"))
    previous = {var.VarName: var.X for var in previous_model.getVars()}

    results = {}
    for label, params, warm, cutoff in variants():
        results[label] = solve(build_variant(paths[day], params, warm, cutoff, previous))
        results[label]["root_gap"] = root_gap(build_variant(paths[day], params, warm, cutoff, previous))
        logger.info("solve_diagnosis.variant", instance=name, variant=label,
                    runtime=results[label]["runtime"], final_gap=results[label]["final_gap"])

    d = np.load(run_path(vector, eps))
    ships = [str(s) for s in d["ships"]]
    recorded = {f"{base}[{s}]": float(d[key][day, j])
                for base, key in (("n_ship_ordered", "ordered"), ("n_ship_sent", "sent"), ("n_ship_discharged", "discharged"))
                for j, s in enumerate(ships)}
    recorded.update({"spot_sold": float(d["spot_sold"][day]), "spot_bought": float(d["spot_bought"][day])})
    out = {"vector": vector, "eps": eps, "day": day, "rollout_solve_s": float(d["solve_s"][day]),
           "rollout_status": str(d["status"][day]), "recorded_day_one": recorded, "variants": results}
    (OUT_DIR / f"{name}.json").write_text(json.dumps(out, indent=1))
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def backfill_root_gaps(vectors):
    """Recompute the root gap of every saved result from its LP and previous-day solution."""
    for vector in vectors:
        for path in sorted(OUT_DIR.glob(f"{vector}-*.json")):
            r = json.loads(path.read_text())
            lp_path = OUT_DIR / f"{path.stem}.lp"
            previous = {}
            for line in (OUT_DIR / f"{instance_name(vector, r['eps'], r['day'] - 1)}.sol").read_text().splitlines():
                if line and not line.startswith("#"):
                    name, val = line.split()
                    previous[name] = float(val)
            for label, params, warm, cutoff in variants():
                r["variants"][label]["root_gap"] = root_gap(build_variant(lp_path, params, warm, cutoff, previous))
            path.write_text(json.dumps(r, indent=1))
            logger.info("solve_diagnosis.root_gaps", instance=path.stem)


def same_decisions(a, b):
    keys = set(a) | set(b)
    return all(abs(a.get(k, 0.0) - b.get(k, 0.0)) <= 1e-3 * max(1.0, abs(a.get(k, 0.0))) for k in keys)


def report(vectors, epsilons=EPSILONS):
    for vector in vectors:
        print(f"\n=== {vector} ===")
        print(f"  {'instance':<20} {'variant':<6} {'root gap':>9} {'to 0.5%':>12} {'best@10s':>10} {'best@120s':>10} "
              f"{'nodes':>8}  day-1 decisions = E0?")
        for path in sorted(OUT_DIR.glob(f"{vector}-*.json"), key=lambda p: (p.stem.split("-")[1], int(p.stem.split("day")[1]))):
            r = json.loads(path.read_text())
            e0 = r["variants"]["E0"]["decisions"]
            print(f"  {path.stem:<20} rollout {r['rollout_solve_s']:.1f} s ({r['rollout_status']}); "
                  f"E0 day-1 = recorded: {same_decisions(e0, r['recorded_day_one'])}")
            for label, v in r["variants"].items():
                root = f"{100 * v['root_gap']:.2f}%" if v["root_gap"] is not None else "n/a"
                reach = f"{v['gap_time']:.1f} s" if v["gap_time"] is not None else f"t/o {100 * v['final_gap']:.2f}%"
                early = f"{v['best_early']:.3f}" if v["best_early"] is not None else "none"
                print(f"  {'':<20} {label:<6} {root:>9} {reach:>12} {early:>10} {v['best_final']:>10.3f} "
                      f"{int(v['nodes']):>8}  {same_decisions(v['decisions'], e0)}")
        clustering(vector, epsilons)


def clustering(vector, epsilons=EPSILONS):
    """Months, half-spreads and buyer tank fill of the slow solves against all solves."""
    print(f"\n  {vector}: where the slow solves sit (slowest {N_SLOWEST}, and all solves >= 10 s), against all days")
    days = []
    for eps in epsilons:
        d = np.load(run_path(vector, eps))
        lower, upper = float(d["inventory_lower"]), float(d["inventory_upper"])
        fill = (d["buyer_inventory_start"] - lower) / (upper - lower)
        days += [(float(d["solve_s"][i]), eps, int(d["month"][i]), float(fill[i]), str(d["status"][i])) for i in range(len(d["day"]))]
    days = np.array(days, dtype=object)
    slowest = sorted(days.tolist(), key=lambda r: -r[0])[:N_SLOWEST]
    slow = [r for r in days.tolist() if r[0] >= 10]

    def summary(rows):
        rows = list(rows)
        months = np.bincount([r[2] for r in rows], minlength=13)[1:]
        fill = np.array([r[3] for r in rows])
        return (len(rows), {eps: sum(r[1] == eps for r in rows) for eps in epsilons}, months.tolist(),
                f"{np.mean(fill < 0.1):.0%}", f"{np.mean(fill > 0.9):.0%}", f"{np.median(fill):.2f}")

    for name, rows in (("slowest", slowest), (">= 10 s", slow), ("all days", days.tolist())):
        n, by_eps, months, empty, full, median = summary(rows)
        print(f"   {name:<9} n={n:<4} by eps {by_eps} | by month {months} | tank <10% full {empty}, >90% {full}, median fill {median}")


def main(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vectors = [args.vector] if args.vector is not None else sorted(lr.DESIGNS)
    epsilons = [args.eps] if args.eps is not None else EPSILONS
    if args.list:
        for i, (vector, eps, day, solve_s) in enumerate(select_instances(vectors, epsilons)):
            print(f"{i:>2} {vector} eps={eps:g} day={day} rollout solve {solve_s:.1f} s")
        return
    if args.report_only:
        report(vectors, epsilons)
        return
    if args.root_gaps:
        backfill_root_gaps(vectors)
        return
    configure_logging(log_file=OUT_DIR / f"diagnosis-worker{args.worker}.log")
    instances = select_instances(vectors, epsilons)
    for vector, eps, day, _ in instances[args.worker::args.n_workers]:
        if (OUT_DIR / f"{instance_name(vector, eps, day)}.json").exists():
            continue
        run_instance(vector, eps, day)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose slow logistics solves.")
    parser.add_argument("--list", action="store_true", help="List the selected instances.")
    parser.add_argument("--report_only", action="store_true", help="Report the saved results.")
    parser.add_argument("--root_gaps", action="store_true", help="Recompute root gaps of the saved results.")
    parser.add_argument("--vector", type=str, default=None, choices=sorted(lr.DESIGNS), help="Only this carrier.")
    parser.add_argument("--eps", type=float, default=None, help="Only this half-spread.")
    parser.add_argument("--worker", type=int, default=0, help="Index of this worker.")
    parser.add_argument("--n_workers", type=int, default=1, help="Number of workers.")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
