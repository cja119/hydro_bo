"""Single-node Bayesian optimisation of one archetype problem (design capacities + contract factor), maximising mean
NPV (MeanVarBayesopt with lambda = 0; the log-variance GP models the seed noise).

The BO is seeded with every Sobol result of the problem and every earlier BO iteration found on disk, then proposes one
point at a time and evaluates it on all seeds in parallel. Each iteration is written as soon as it finishes
(<bo.out_dir>/<problem>/iter_<k>.json), and the job stops starting new iterations `walltime_buffer_s` before its
walltime, so a resubmitted job carries on where the last one stopped.

    python scripts/archetype_bo.py --problem medium-NH3 --walltime-h 72
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from hydro_bo.archetypes.pipeline import (  # noqa: E402
    evaluate_point, load_config, load_results, ncpus_default, problem_by_key, write_json)
from hydro_bo.archetypes.scenarios import SEARCH_KEYS  # noqa: E402


def _walltime_s(args) -> float:
    if args.walltime_h is not None:
        return args.walltime_h * 3600
    if os.environ.get("WALLTIME_S"):
        return float(os.environ["WALLTIME_S"])
    return float("inf")


def best_observed(results: list[dict]) -> dict | None:
    ok = [r for r in results if not r.get("failed") and np.isfinite(r.get("mean_npv_bn", np.nan))]
    return max(ok, key=lambda r: r["mean_npv_bn"]) if ok else None


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--problem", required=True, help="e.g. low-NH3, medium-LH2")
    a.add_argument("--config", default=None)
    a.add_argument("--ncpus", type=int, default=None)
    a.add_argument("--walltime-h", type=float, default=None, help="job walltime (default: WALLTIME_S env or none)")
    a.add_argument("--iters", type=int, default=None, help="override bo.iter_budget (total BO iterations)")
    args = a.parse_args()
    t_start = time.time()
    cfg = load_config(args.config)
    problem = problem_by_key(cfg, args.problem)
    ncpus = args.ncpus or ncpus_default()
    budget = args.iters if args.iters is not None else int(cfg["bo"]["iter_budget"])
    walltime, buffer = _walltime_s(args), float(cfg["bo"]["walltime_buffer_s"])
    sobol_dir, bo_dir = ROOT / cfg["sobol"]["out_dir"], ROOT / cfg["bo"]["out_dir"]

    from hydro_bo.utils.jax_threads import configure_jax_threads
    from hydro_bo.utils.run_config import load_config as load_run_config
    run_cfg = load_run_config(ROOT / "scripts" / "config.yml", vector_override=problem.vector, num_devices_override=ncpus)
    configure_jax_threads(run_cfg.nlp.n_devices, run_cfg.nlp.blas_threads)
    from hydro_bo.opt import MeanVarBayesopt  # noqa: E402  (jax import after the thread configuration)

    def _no_direct_eval(x):
        raise RuntimeError("archetype_bo drives evaluation itself")

    bounds = problem.bounds()
    nlp = run_cfg.nlp
    bo = MeanVarBayesopt(
        f=_no_direct_eval, bounds=bounds, n_initial_points=0, iter_limit=0, lam=0.0,
        n_restarts=nlp.acq_n_restarts, pow_sobol=nlp.acq_pow_sobol, seed=int(cfg["sobol"]["seed"]), cat_vars=(),
        sqp_config=nlp.to_sqp_config(), pad_initial=nlp.pad_initial, gp_lbfgs_max_iter=nlp.gp_lbfgs_max_iter,
        gp_mu_kernel=nlp.gp_mu_kernel, gp_log_var_kernel=nlp.gp_log_var_kernel,
    )
    sobol = load_results([sobol_dir], problem.key)
    done = load_results([bo_dir], problem.key)
    for r in sobol + done:
        bo.observe(np.asarray(r["x"], dtype=float), np.asarray(r["samples_bn"], dtype=float))
    print(f"{problem.key}: {len(sobol)} Sobol + {len(done)} earlier BO points loaded; budget {budget}; ncpus {ncpus}",
          flush=True)
    if len(sobol) + len(done) < 2:
        raise SystemExit("need at least 2 observed points to fit the GPs (run the Sobol array first)")

    eval_times = [r["eval_wall_s"] for r in sobol + done if r.get("eval_wall_s")]
    k = len(done)
    while k < budget:
        elapsed = time.time() - t_start
        need = (np.median(eval_times) if eval_times else 1800.0) + 600.0       # one evaluation + acquisition
        if elapsed + need + buffer > walltime:
            print(f"stopping before iteration {k + 1}: {elapsed / 3600:.2f} h used of {walltime / 3600:.2f} h", flush=True)
            break
        t0 = time.time()
        x = np.clip(bo.suggest(), bounds[:, 0], bounds[:, 1])
        acq_s = time.time() - t0
        result = evaluate_point(cfg, problem, x, ncpus, npz_dir=bo_dir / problem.key / f"iter_{k + 1:04d}")
        result.update({"source": "bo", "iteration": k + 1, "acquisition_s": acq_s, "pbs_jobid": os.environ.get("PBS_JOBID")})
        write_json(bo_dir / problem.key / f"iter_{k + 1:04d}.json", result)
        bo.observe(x, np.asarray(result["samples_bn"], dtype=float))
        eval_times.append(result["eval_wall_s"])
        k += 1
        best = best_observed(sobol + load_results([bo_dir], problem.key))
        print(f"iter {k}: mean NPV {result['mean_npv_bn']:.3f} bn (valid {result['n_valid']}), "
              f"best {best['mean_npv_bn']:.3f} bn; acq {acq_s:.0f} s, eval {result['eval_wall_s'] / 60:.1f} min", flush=True)

    results = sobol + load_results([bo_dir], problem.key)
    best = best_observed(results)
    if best is not None:
        summary = {"problem": problem.key, "n_sobol": len(sobol), "n_bo": len(results) - len(sobol),
                   "best_source": best["source"], "best_mean_npv_bn": best["mean_npv_bn"], "best_sd_npv_bn": best["sd_npv_bn"],
                   "best_x": dict(zip(SEARCH_KEYS, best["x"])), "best_design": best["design"]}
        write_json(bo_dir / problem.key / "best.json", summary)
        print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
