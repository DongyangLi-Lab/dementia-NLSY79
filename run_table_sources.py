"""Finish manuscript figure and table data sources after the model run."""
from concurrent.futures import ThreadPoolExecutor
import json
import time

from artifact_store import save_json
from run_benchmark import LANES, gpu_inventory, lane_environment, run_stage
from train.cohorts import ARTIFACTS, sha256


def main():
    status = ARTIFACTS / "source_status.json"
    model_status = ARTIFACTS / "run_status.json"
    save_json(status, {"state": "waiting_for_benchmark"})
    try:
        while True:
            model = json.loads(model_status.read_text(encoding="utf-8"))
            if model["state"] == "failed":
                raise RuntimeError("Benchmark failed. See run_status.json and the lane logs.")
            if model["state"] == "complete":
                break
            time.sleep(30)
        gpus = gpu_inventory()
        cpu = lane_environment("cpu", gpus, 0)
        save_json(status, {"state": "running_imputation_comparison"})
        with ThreadPoolExecutor(max_workers=3) as pool:
            jobs = [pool.submit(run_stage, f"imputation-comparison-{device}", "train.imputation_comparison",
                                ("--device", device, "--models", *models), lane_environment(device, gpus, 0))
                    for device, models in LANES.items()]
            for job in jobs:
                job.result()
        run_stage("merge-imputation-comparison", "train.imputation_comparison", ("--merge",), cpu)
        run_stage("predictor-tables", "inference.predictor_tables", (), cpu)
        run_stage("final-table-sources", "inference.supplementary_tables", (), cpu)
        tables = ARTIFACTS / "tables"
        expected = ["Table1", "S1", *(f"S{n}" for n in range(9, 22)), "S22a", "S22b", "S23",
                    "S24a", "S24b", "S25", "S26", "S27"]
        assert all((tables / "supplementary" / f"{name}.csv").is_file() for name in expected)
        save_json(status, {"state": "complete", "table_files": {
            p.relative_to(ARTIFACTS).as_posix(): sha256(p) for p in sorted(tables.rglob("*.csv"))}})
    except Exception as error:
        save_json(status, {"state": "failed", "error": str(error)})
        raise


if __name__ == "__main__":
    main()
