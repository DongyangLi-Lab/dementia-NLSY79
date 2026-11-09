"""Run the revised benchmark on CPU and two local GPUs with resumable stages."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import time

from artifact_store import save_json
from train.cohorts import ARTIFACTS, ROOT, TRACKER
from train.prepared import COMBINATIONS, PREPARED_RECEIPT, cache_files

LANES = {"cpu": ("LR", "GNB", "DT"), "0": ("LightGBM", "TabICL"), "1": ("CatBoost", "RealMLP", "GAT")}
LOGS = ARTIFACTS / "logs"


def gpu_inventory():
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"], text=True)
    return {i: (uuid, name) for i, uuid, name in (line.strip().split(", ", 2) for line in result.splitlines())}


def run_stage(name, module, args, env, python=sys.executable):
    log = LOGS / f"{name}.log"
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"\n[{datetime.now(timezone.utc).isoformat()}] {module} {' '.join(map(str,args))}\n")
        stream.flush()
        proc = subprocess.run([str(python), "-u", "-m", module, *map(str, args)],
                              cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if proc.returncode:
        raise RuntimeError(f"{name}: exit {proc.returncode}, see {log}")
    print(f"{name}: complete", flush=True)


def lane_environment(device, gpus, opencl_device):
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="2",
               LOKY_MAX_CPU_COUNT=str(os.cpu_count()), CUDA_DEVICE_ORDER="PCI_BUS_ID",
               CUDA_VISIBLE_DEVICES="" if device == "cpu" else gpus[device][0],
               HRS_OPENCL_DEVICE=str(opencl_device), HF_HUB_OFFLINE="1")
    return env


def prepare_inputs(args, cpu_env):
    PREPARED_RECEIPT.unlink(missing_ok=True)
    if args.wait_existing_imputers:
        deadline = time.monotonic() + 3600 * args.cache_timeout_hours
        paths = cache_files()
        while missing := [name for name, path in paths.items() if not path.is_file()]:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Preprocessing still missing: {', '.join(missing)}")
            time.sleep(15)
    else:
        run_stage("cohorts", "train.cohorts", ("--tracker", args.tracker.resolve()), cpu_env)
        with ThreadPoolExecutor(max_workers=len(COMBINATIONS)) as pool:
            pending = [
                pool.submit(
                    run_stage, f"impute-{outcome}-h{horizon}", "train.imputation",
                    ("--horizon", horizon, "--outcome", outcome), dict(cpu_env, HRS_THREADS="3"),
                    args.imputer_python.resolve(),
                )
                for horizon, outcome in COMBINATIONS
            ]
            for future in as_completed(pending):
                future.result()
    run_stage("prepare-verification", "inference.verify", ("--prepared-only",), cpu_env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker", type=Path, default=TRACKER)
    parser.add_argument("--imputer-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--opencl-device", type=int, default=0,
                        help="OpenCL index for the NVIDIA GPU 0, verified separately")
    parser.add_argument("--wait-existing-imputers", action="store_true",
                        help="Consume caches from four already running imputation jobs")
    parser.add_argument("--prepare-only", action="store_true", help="Finish and verify data without launching model workers")
    parser.add_argument("--cache-timeout-hours", type=float, default=12)
    args = parser.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    gpus = {} if args.prepare_only else gpu_inventory()
    if not args.prepare_only and not {"0", "1"} <= gpus.keys():
        raise RuntimeError("Physical GPUs 0 and 1 must be available")
    state_path = ARTIFACTS / "run_status.json"
    lock_path = ARTIFACTS / "benchmark.lock"
    with lock_path.open("x", encoding="utf-8") as lock:
        lock.write(str(os.getpid()))
    statuses = {}
    try:
        save_json(state_path, {"state": "preparing", "pid": os.getpid(), "gpus": gpus})
        cpu_env = lane_environment("cpu", gpus, args.opencl_device)
        prepare_inputs(args, cpu_env)
        if args.prepare_only:
            save_json(state_path, {"state": "prepared", "imputations": len(cache_files())})
            return
        save_json(state_path, {"state": "running", "phase": "training", "pid": os.getpid(),
                               "imputations": len(cache_files()), "gpus": gpus})
        with ThreadPoolExecutor(max_workers=len(LANES)) as pool:
            lanes = {}
            for device, models in LANES.items():
                env = lane_environment(device, gpus, args.opencl_device)
                task = pool.submit(run_stage, f"lane-{device}", "train.lane",
                                   ("--device", device, *models), env)
                lanes[task] = device
            for future in as_completed(lanes):
                device = lanes[future]
                try:
                    future.result()
                    statuses[device] = []
                except Exception as error:
                    statuses[device] = [str(error)]
                save_json(state_path, {"state": "running", "phase": "training", "pid": os.getpid(),
                                       "imputations": len(cache_files()), "gpus": gpus, "lanes": statuses})
        if any(statuses.values()):
            raise RuntimeError("Some model stages failed")
        run_stage("verify", "inference.verify", (), cpu_env)
        for outcome in ("final", "any"):
            run_stage(f"metrics-{outcome}", "inference.evaluation", ("--outcome", outcome), cpu_env)
        run_stage("cohort-tables", "inference.cohort_tables", (), cpu_env)
        with ThreadPoolExecutor(max_workers=3) as pool:
            def importance_lane(device, models):
                env = lane_environment(device, gpus, args.opencl_device)
                for model in models:
                    run_stage(f"importance-{model}", "inference.importance", ("--model", model), env)
            pending = [pool.submit(importance_lane, device, models) for device, models in LANES.items()]
            for future in as_completed(pending):
                future.result()
        run_stage("merge-importance", "inference.importance", ("--merge",), cpu_env)
        run_stage("table-sources", "inference.supplementary_tables", (), cpu_env)
        save_json(state_path, {"state": "complete", "gpus": gpus, "lanes": statuses})
    except Exception as error:
        save_json(state_path, {"state": "failed", "gpus": gpus, "lanes": statuses, "error": str(error)})
        raise
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
