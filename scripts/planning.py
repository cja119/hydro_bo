
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hydro_bo import Planning, configure_logging
from src.algs.logging_config import get_logger

configure_logging()
logger = get_logger(__name__)


def run_planning(vector_type: str):
    tmp_dir = Path(__file__).parent / "tmp"
    environment = Planning(
        f"{vector_type}-Chile", weather_file="CoastalChile_15-20_Wind.csv", tmp_dir=tmp_dir
    )

    with environment as env:
        env["booleans"]["vector_choice"][vector_type] = True
        env["booleans"]["electrolysers"]["SOFC"] = True
        env["booleans"]["wind"] = True
        env["equipment"]["vector_production"][vector_type] = (
            4 if vector_type == "NH3" else 12
        )

    environment.solve()
    environment.get_results()


if __name__ == "__main__":
    assert sys.argv[1] in ["NH3", "LH2"], "Please provide a valid vector type: NH3 or LH2"

    logger.info("planning.start", vector=sys.argv[1])
    run_planning(sys.argv[1])
    logger.info("planning.complete", vector=sys.argv[1], results_path=f"scripts/tmp/planning/{sys.argv[1]}-Chile.yml")