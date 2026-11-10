"""Sample resource use from an already running local GPU lane."""
import argparse
import json
import subprocess
import time

import psutil

from artifact_store import save_json
from cluster.local_dispatch import process_sample
from run_benchmark import gpu_inventory
from train.cohorts import ARTIFACTS


def monitor(device, pid):
    proc = psutil.Process(pid)
    if "train.lane" not in proc.cmdline() or proc.cmdline()[proc.cmdline().index("--device") + 1] != device:
        raise ValueError("Expected the existing benchmark lane")
    uuid, gpu = gpu_inventory()[device]
    status = ARTIFACTS / "logs" / f"lane-{device}-status.json"
    current = None
    rss = vram = 0
    samples = 0
    start = time.perf_counter()
    def save():
        if current is not None:
            outcome, h, model = current
            save_json(ARTIFACTS / "runtime" / f"sampled-{outcome}-h{h}-{model}.json", {
                "outcome": outcome, "horizon": h, "model": model, "device": gpu, "site": "local",
                "wall_sampled_seconds": time.perf_counter() - start,
                "sampled_peak_process_tree_rss_bytes": rss,
                "sampled_peak_device_used_mib": vram if samples else None,
                "sampled_device_memory_scope": "whole GPU including other allocations",
                "sampling_interval_seconds": 5, "mode": "observational sampling of an existing lane",
            })
    while proc.is_running():
        jobs = json.loads(status.read_text(encoding="utf-8"))["pending"]
        if not jobs:
            break
        key = tuple(jobs[0])
        if key != current:
            save()
            current = key
            rss = vram = samples = 0
            start = time.perf_counter()
        rss = max(rss, process_sample(proc))
        try:
            row = subprocess.check_output(["nvidia-smi", "-i", uuid, "--query-gpu=memory.used",
                                           "--format=csv,noheader,nounits"], text=True, timeout=5)
            vram = max(vram, int(row.strip()))
            samples += 1
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        time.sleep(5)
    save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("device")
    parser.add_argument("pid", type=int)
    args = parser.parse_args()
    monitor(args.device, args.pid)
