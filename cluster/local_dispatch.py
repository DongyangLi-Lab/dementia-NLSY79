"""Continue the local benchmark after delegating selected jobs to UNC."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from artifact_store import job_dir, save_json
from cluster.model_dispatch import REMOTE
from model.specs import MODELS
from run_benchmark import LANES, LOGS, gpu_inventory, lane_environment, run_stage
from train.cohorts import ARTIFACTS, ROOT
from train.prepared import COMBINATIONS, require_prepared

DISPATCH = ARTIFACTS / "unc" / "model_dispatch.json"
STATE = ARTIFACTS / "run_status.json"
LOCK = ARTIFACTS / "benchmark.lock"


def checkpoints():
    return set((ARTIFACTS / "checkpoints").glob("*/*/*/rep*.done.json"))


def takeover():
    owner = psutil.Process(int(LOCK.read_text()))
    if not any(Path(arg).name == "run_benchmark.py" for arg in owner.cmdline()):
        raise RuntimeError("The existing coordinator is not the expected benchmark process")
    lanes = {p.cmdline()[p.cmdline().index("--device") + 1]: p for p in owner.children()
             if "train.lane" in p.cmdline() and "--device" in p.cmdline()}
    gpu1 = lanes["1"]
    before = checkpoints()
    deadline = time.monotonic() + 360
    while time.monotonic() < deadline:
        new = checkpoints() - before
        if any(p.parent.name in LANES["1"] for p in new):
            break
        if not gpu1.is_running():
            raise RuntimeError("GPU 1 worker exited before the planned takeover")
        time.sleep(1)
    else:
        raise TimeoutError("GPU 1 did not reach a saved repeat within six minutes")

    for outcome, h, model in REMOTE["benchmark"]:
        path = job_dir("checkpoints", h, outcome, model)
        if path.exists() and any(path.iterdir()):
            raise RuntimeError(f"Task already started locally: {path}")
    descendants = [p for p in gpu1.children(recursive=True) if "train.lane" in p.cmdline()]
    owner.kill()
    owner.wait(10)
    for process in descendants:
        if process.is_running():
            process.kill()
    if not descendants and gpu1.is_running():
        gpu1.kill()
    gpu1.wait(20)
    LOCK.unlink(missing_ok=True)
    return {dev: p for dev, p in lanes.items() if dev != "1"}


def process_sample(process):
    rss = 0
    try:
        children = process.children(recursive=True)
    except psutil.Error:
        children = []
    for item in [process, *children]:
        try:
            rss += item.memory_info().rss
        except psutil.Error:
            pass
    return rss


def run_local_gpu(task, gpus):
    outcome, horizon, model = task
    env = lane_environment("1", gpus, 0)
    flags = ["--horizon", str(horizon), "--outcome", outcome, "--model", model]
    name = f"local-{outcome}-h{horizon}-{model}"
    profile = ARTIFACTS / "runtime" / f"{name}.json"
    start = time.perf_counter()
    rss = 0
    vram = []
    used_gpu = gpus["1"][0]
    with (LOGS / f"{name}.log").open("a", encoding="utf-8") as log:
        for module, extra in (("train.benchmark", ["--device", "1", "--threads", "2"]), ("inference.predict", [])):
            child = subprocess.Popen([sys.executable, "-u", "-m", module, *flags, *extra],
                                     cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            proc = psutil.Process(child.pid)
            while child.poll() is None:
                rss = max(rss, process_sample(proc))
                try:
                    row = subprocess.check_output(
                        ["nvidia-smi", "-i", used_gpu, "--query-gpu=memory.used",
                         "--format=csv,noheader,nounits"], text=True, timeout=5)
                    vram.append(int(row.strip()))
                except (OSError, ValueError, subprocess.SubprocessError):
                    pass
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    continue
            if child.returncode:
                raise RuntimeError(f"{name} failed; see {log.name}")
    save_json(profile, {"site": "local", "model": model, "outcome": outcome, "horizon": horizon,
                        "device": gpus["1"][1], "wall_seconds": time.perf_counter() - start,
                        "sampled_peak_process_tree_rss_bytes": rss,
                        "sampled_peak_device_used_mib": max(vram) if vram else None,
                        "sampled_device_memory_scope": "whole GPU including other allocations",
                        "sampling_interval_seconds": 5, "mode": "resumed task, if checkpoints existed"})


def ready_benchmark():
    return all((job_dir("predictions", h, o, m) / "inference.done.json").is_file()
               for h, o in COMBINATIONS for m in MODELS)


def main():
    require_prepared()
    gpus = gpu_inventory()
    keep = takeover()
    with LOCK.open("x", encoding="utf-8") as lock:
        lock.write(str(os.getpid()))
    remote = {tuple(task) for task in REMOTE["benchmark"]}
    local = [(o, h, m) for h, o in COMBINATIONS for m in LANES["1"] if (o, h, m) not in remote]
    status = {"state": "running", "phase": "distributed_training", "pid": os.getpid(),
              "gpus": gpus, "imputations": 44}
    try:
        save_json(STATE, status)
        save_json(DISPATCH, {"ready_to_submit": True, "unc": REMOTE, "local_gpu1": local,
                             "local_preserved_workers": {k: p.pid for k, p in keep.items()},
                             "created_utc": datetime.now(timezone.utc).isoformat()})
        for task in local:
            # Finished local assignments already have their timing records.
            o, h, model = task
            if not (job_dir("predictions", h, o, model) / "inference.done.json").is_file():
                run_local_gpu(task, gpus)
        deadline = time.monotonic() + 24 * 3600
        while not ready_benchmark():
            if time.monotonic() >= deadline:
                raise TimeoutError("Some distributed benchmark tasks did not return within 24 hours")
            for dev, worker in keep.items():
                lane_state = json.loads((LOGS / f"lane-{dev}-status.json").read_text(encoding="utf-8"))
                if not worker.is_running() and (lane_state["pending"] or lane_state["failed"]):
                    raise RuntimeError(f"Preserved lane {dev} exited with incomplete assignments")
            time.sleep(20)
        cpu_env = lane_environment("cpu", gpus, 0)
        run_stage("verify-distributed", "inference.verify", (), cpu_env)
        for outcome in ("final", "any"):
            run_stage(f"metrics-{outcome}", "inference.evaluation", ("--outcome", outcome), cpu_env)
        run_stage("cohort-tables", "inference.cohort_tables", (), cpu_env)
        with ThreadPoolExecutor(max_workers=len(LANES)) as pool:
            def importance_lane(device, models):
                env = lane_environment(device, gpus, 0)
                for model in models:
                    run_stage(f"importance-{model}", "inference.importance", ("--model", model), env)
            futures = [pool.submit(importance_lane, dev, models) for dev, models in LANES.items()]
            for future in as_completed(futures):
                future.result()
        run_stage("merge-importance", "inference.importance", ("--merge",), cpu_env)
        run_stage("table-sources", "inference.supplementary_tables", (), cpu_env)
        save_json(STATE, {**status, "state": "complete"})
    except Exception as error:
        save_json(STATE, {**status, "state": "failed", "error": str(error)})
        raise
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
