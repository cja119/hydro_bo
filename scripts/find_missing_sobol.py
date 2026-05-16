"""Print Sobol row indices that were submitted but never wrote a result.

Walks `<sobol_dir>/row_NNNNN/` directories under the configured `sobol.sobol_dir`
for the given vector and prints (one per line, stdout) the indices of rows that
have no `result_*.json` file. Pipe to a file and feed to `jobs/sobol_topup.sh`.

Usage:
    python scripts/find_missing_sobol.py LH2 > missing_lh2.txt
    python scripts/find_missing_sobol.py NH3 > missing_nh3.txt
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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

    missing = []
    for child in sorted(sobol_base.iterdir()):
        m = ROW_RE.match(child.name)
        if not m or not child.is_dir():
            continue
        idx = int(m.group(1))
        if not any(child.glob("result_*.json")):
            missing.append(idx)

    print(f"scanned {sobol_base}: {len(missing)} missing", file=sys.stderr)
    for idx in missing:
        print(idx)


if __name__ == "__main__":
    main()
