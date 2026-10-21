"""Throughput profile for one node: keep `ncpus` single-seed rollouts in flight for `--minutes`, cycling over every
archetype problem with designs from each problem's Sobol rows, and report rollouts per hour, rollout time per problem,
peak memory, and what the Sobol array and the BO would cost at that pace.

    python scripts/archetype_profile.py --minutes 60            # on the compute node (ncpus from NCPUS)
    python scripts/archetype_profile.py --minutes 5 --ncpus 4   # quick local check

Results: <sobol.out_dir>/../archetype_profile/profile_<jobid or time>.json, plus a printed summary.
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from hydro_bo.archetypes.pipeline import load_config, ncpus_default, problems, seeds, task_point, write_json  # noqa: E402
from hydro_bo.archetypes.rollout import run_seed  # noqa: E402
from hydro_bo.archetypes.scenarios import design_from_x  # noqa: E402


class MemorySampler(threading.Thread):
    """Peak resident memory of this process and its children, and the lowest available system memory [GB]."""

    def __init__(self, every_s: float = 5.0):
        super().__init__(daemon=True)
        self.every_s, self.peak_gb, self.min_avail_gb, self.peak_children = every_s, 0.0, float("inf"), 0
        self._stop = threading.Event()

    def run(self):
        import psutil
        me = psutil.Process()
        while not self._stop.is_set():
            try:
                procs = [me] + me.children(recursive=True)
                rss = sum(p.memory_info().rss for p in procs if p.is_running())
                self.peak_gb = max(self.peak_gb, rss / 1024**3)
                self.peak_children = max(self.peak_children, len(procs) - 1)
                self.min_avail_gb = min(self.min_avail_gb, psutil.virtual_memory().available / 1024**3)
            except Exception:
                pass
            self._stop.wait(self.every_s)

    def stop(self):
        self._stop.set()


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--config", default=None)
    a.add_argument("--ncpus", type=int, default=None)
    a.add_argument("--minutes", type=float, default=60.0)
    a.add_argument("--drain", action="store_true", help="wait for rollouts still running at the deadline")
    args = a.parse_args()
    cfg = load_config(args.config)
    ncpus = args.ncpus or ncpus_default()
    probs = problems(cfg)
    n_rows = int(cfg["sobol"]["n_per_problem"])
    seed_list = seeds(cfg)

    def job(k: int):
        """k-th rollout: problems in turn, Sobol rows in turn, seeds in turn."""
        p_idx, j = k % len(probs), k // len(probs)
        problem, _, _, x = task_point(cfg, p_idx * n_rows + (j % n_rows))
        return problem.key, (design_from_x(x, problem), problem, seed_list[j % len(seed_list)], None)

    mem = MemorySampler()
    mem.start()
    t0 = time.time()
    deadline = t0 + args.minutes * 60
    done, running, k = [], {}, 0
    with ProcessPoolExecutor(max_workers=ncpus, mp_context=get_context("spawn")) as pool:
        while True:
            while time.time() < deadline and len(running) < ncpus:
                key, jargs = job(k)
                running[pool.submit(run_seed, *jargs)] = (key, time.time())
                k += 1
            if not running or (time.time() >= deadline and not args.drain):
                break
            finished, _ = wait(list(running), timeout=30, return_when=FIRST_COMPLETED)
            for fut in finished:
                key, started = running.pop(fut)
                r = fut.result()
                done.append({"problem": key, "wall_s": r.get("wall_s"), "error": r.get("error"),
                             "npv_bn": (r.get("npv") or float("nan")) / 1000, "finished_s": time.time() - t0,
                             "time_limited_days": r.get("time_limited_days")})
                print(f"[{(time.time() - t0) / 60:6.1f} min] {key:<11} {r.get('wall_s', 0) / 60:5.1f} min "
                      f"{'ERROR ' + str(r.get('error')) if r.get('error') else ''}", flush=True)
        for fut in running:
            fut.cancel()
    elapsed_h = (time.time() - t0) / 3600
    mem.stop()

    ok = [d for d in done if not d["error"]]
    per_problem = {}
    for p in probs:
        w = [d["wall_s"] for d in ok if d["problem"] == p.key]
        per_problem[p.key] = {"n": len(w), "mean_min": float(np.mean(w) / 60) if w else None,
                              "p90_min": float(np.percentile(w, 90) / 60) if w else None,
                              "max_min": float(np.max(w) / 60) if w else None}
    rate = len(ok) / elapsed_h if elapsed_h > 0 else 0.0
    n_seeds, n_sobol, iters = len(seed_list), n_rows * len(probs), int(cfg["bo"]["iter_budget"])
    waves = -(-n_seeds // ncpus)                    # a point's seeds run in waves; each wave waits for its slowest rollout
    for v in per_problem.values():
        v["hours_per_point"] = waves * v["p90_min"] / 60 if v["p90_min"] else None
    by_problem = [v["hours_per_point"] for v in per_problem.values() if v["hours_per_point"]]
    point_h = float(np.mean(by_problem)) if by_problem else float("nan")
    report = {
        "ncpus": ncpus, "minutes": args.minutes, "elapsed_h": elapsed_h, "rollouts_completed": len(ok),
        "rollouts_failed": len(done) - len(ok), "rollouts_per_hour": rate,
        "rollouts_per_core_hour": rate / ncpus, "per_problem": per_problem,
        "peak_memory_gb": mem.peak_gb, "min_available_memory_gb": mem.min_avail_gb, "peak_child_processes": mem.peak_children,
        "projection": {
            "seeds_per_point": n_seeds, "waves_per_point": waves, "hours_per_point_on_node": point_h,
            "sobol_points": n_sobol, "sobol_node_hours": n_sobol * point_h,
            "bo_iterations_per_problem": iters, "bo_hours_per_problem": iters * point_h,
        },
        "pbs_jobid": os.environ.get("PBS_JOBID"), "rollouts": done,
    }
    out_dir = (ROOT / cfg["sobol"]["out_dir"]).parent / "archetype_profile"
    tag = os.environ.get("PBS_JOBID", time.strftime("%Y%m%d_%H%M%S")).split(".")[0]
    write_json(out_dir / f"profile_{tag}.json", report)
    print("\n" + json.dumps({k: v for k, v in report.items() if k != "rollouts"}, indent=1, default=float))


if __name__ == "__main__":
    main()
