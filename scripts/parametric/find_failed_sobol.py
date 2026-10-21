"""Print parametric-Sobol row indices that need re-running.

Walks `<sobol_dir>/row_NNNNN/` under the parametric `sobol.sobol_dir` for the
given vector and prints (one index per line, stdout) every row that either

  * has no `result_*.json` at all, or
  * whose newest `result_*.json` has no usable objective — `objective` is null
    or non-finite. This covers both a crashed dispatcher (status != "complete")
    AND a dispatcher that ran to "completion" but had every worker fail
    (`n_failed == n_workers`, so `objective` is null). NOTE: `status ==
    "complete"` alone does NOT mean success — it only means the dispatcher
    didn't crash; the real success signal is a finite `objective`.

Rows with a finite objective are kept, even if it is very negative or some
individual workers were NaN — those are genuine solves that merely scored low.

Pipe to a file and feed to `jobs/parametric_sobol_topup.sh`:

    python scripts/parametric/find_failed_sobol.py NH3 > failed_param_nh3.txt
    ./jobs/parametric_sobol_topup.sh NH3 failed_param_nh3.txt
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hydro_bo.utils.run_config import load_config, resolve_sobol_dir

SCRIPTS_DIR = Path(__file__).resolve().parent
ROW_RE = re.compile(r"^row_(\d+)$")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("vector", type=str, help="Hydrogen vector, e.g. LH2 or NH3.")
    args = parser.parse_args()

    cfg = load_config(SCRIPTS_DIR / "config.yml", vector_override=args.vector)
    sobol_base = resolve_sobol_dir(cfg.sobol.sobol_dir, SCRIPTS_DIR, cfg.general.vector)
    if sobol_base is None or not sobol_base.exists():
        print(f"sobol_dir not found: {sobol_base}", file=sys.stderr)
        sys.exit(1)

    failed, no_result = [], 0
    for child in sorted(sobol_base.iterdir()):
        m = ROW_RE.match(child.name)
        if not m or not child.is_dir():
            continue
        idx = int(m.group(1))
        results = sorted(child.glob("result_*.json"))
        if not results:
            failed.append(idx)
            no_result += 1
            continue
        try:
            obj = json.load(open(results[-1])).get("objective")
        except (json.JSONDecodeError, OSError):
            obj = None
        # Re-run if there is no usable objective: null (crashed, or all workers
        # failed) or non-finite (sentinel penalty / NaN).
        if obj is None or not math.isfinite(float(obj)):
            failed.append(idx)

    print(
        f"scanned {sobol_base}: {len(failed)} to re-run "
        f"({no_result} with no result, {len(failed) - no_result} with no usable objective)",
        file=sys.stderr,
    )
    for idx in failed:
        print(idx)


if __name__ == "__main__":
    main()
