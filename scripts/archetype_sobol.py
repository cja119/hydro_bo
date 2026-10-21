"""One PBS array task of the archetype Sobol pool: evaluate one (problem, Sobol row) design with every seed.

Array index i maps to problem i // n_per_problem and Sobol row i % n_per_problem (problem order = archetypes x vectors
as listed in scripts/config_archetypes.yml). Results go to <sobol.out_dir>/<problem>/row_<row>.json; an existing result
is skipped unless --force, so the array can simply be resubmitted to fill gaps.

    python scripts/archetype_sobol.py --list                     # tasks, problems, trains and bounds
    python scripts/archetype_sobol.py --index 5 --dry-run        # show the design only
    python scripts/archetype_sobol.py                            # index from PBS_ARRAY_INDEX, ncpus from NCPUS
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hydro_bo.archetypes.pipeline import (  # noqa: E402
    evaluate_point, load_config, ncpus_default, problems, task_point, write_json)
from hydro_bo.archetypes.scenarios import SEARCH_KEYS, design_from_x  # noqa: E402


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--config", default=None)
    a.add_argument("--index", type=int, default=None, help="array index (default: PBS_ARRAY_INDEX)")
    a.add_argument("--ncpus", type=int, default=None, help="parallel rollouts (default: NCPUS / os.cpu_count)")
    a.add_argument("--list", action="store_true")
    a.add_argument("--dry-run", action="store_true")
    a.add_argument("--force", action="store_true")
    args = a.parse_args()
    cfg = load_config(args.config)
    n = int(cfg["sobol"]["n_per_problem"])
    probs = problems(cfg)

    if args.list:
        print(f"{n * len(probs)} array tasks: {len(probs)} problems x {n} Sobol rows (PBS -J 0-{n * len(probs) - 1})")
        for i, p in enumerate(probs):
            print(f"  [{i * n}-{(i + 1) * n - 1}] {p.key}: {p.trains} trains x {p.train_size:.3f} t/h, nu {p.archetype.premium},"
                  f" eps {p.archetype.half_spread}, depth {p.archetype.depth_days} d")
            for k, (lo, hi) in zip(SEARCH_KEYS, p.bounds()):
                print(f"      {k:<28} [{lo:12.3f}, {hi:12.3f}]")
        return

    index = args.index if args.index is not None else int(os.environ.get("PBS_ARRAY_INDEX", os.environ.get("PBS_ARRAYID", -1)))
    problem, row, unit, x = task_point(cfg, index)
    out = ROOT / cfg["sobol"]["out_dir"] / problem.key / f"row_{row:04d}.json"
    print(f"task {index}: {problem.key} row {row} -> {out}", flush=True)
    if args.dry_run:
        for k, v in design_from_x(x, problem).items():
            print(f"  {k}: {v}")
        return
    if out.exists() and not args.force:
        print("result exists; skipping (use --force to re-run)")
        return
    ncpus = args.ncpus or ncpus_default()
    result = evaluate_point(cfg, problem, x, ncpus, npz_dir=out.with_suffix(""))
    result.update({"source": "sobol", "row": row, "unit_sample": [float(u) for u in unit], "array_index": index,
                   "pbs_jobid": os.environ.get("PBS_JOBID")})
    write_json(out, result)
    print(f"done: mean NPV {result['mean_npv_bn']:.3f} bn, sd {result['sd_npv_bn']:.3f}, valid {result['n_valid']}/"
          f"{len(result['seeds'])}, {result['eval_wall_s'] / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
