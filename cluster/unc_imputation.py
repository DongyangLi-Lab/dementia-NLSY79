"""Run an explicitly assigned subset of HRS imputations on a Slurm allocation."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
import pickle
from pathlib import Path
import socket
import time
from zipfile import ZIP_DEFLATED, ZipFile

from train import imputation as imp
from train.cohorts import ARTIFACTS, load_cohorts, sha256
from train.prepared import assert_predictors


def compute(task):
    horizon, outcome, stage = task
    stamp = imp.cache_stamp(horizon, outcome)
    path = imp.cache_path(horizon, outcome, stage)
    dev, ext, columns = load_cohorts(horizon, outcome)
    X = dev[columns].astype(float)
    if path.exists():
        imp.read_cache(horizon, outcome, stage, stamp)
        return path
    started = time.perf_counter()
    if stage == "external":
        fit, (temporal,) = imp._impute(X, [ext[columns].astype(float)], seed=999)
        payload = {"X_dev": fit, "X_ext": temporal}
    else:
        r = int(stage.removeprefix("rep"))
        tr, va, te = imp.split_indices(dev.y.to_numpy(), r)
        fit, (val, test) = imp._impute(X.iloc[tr], [X.iloc[va], X.iloc[te]], seed=42+r)
        payload = {"train": tr, "val": va, "test": te, "X_train": fit, "X_val": val, "X_test": test}
    seconds = time.perf_counter() - started
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as stream:
        pickle.dump({**payload, "stamp": stamp, "predictors": columns, "imputer_fits": 1,
                     "seconds": seconds, "device": "CPU"}, stream, protocol=5)
    temp.replace(path)
    print(f"{outcome}/h{horizon}/{stage} complete in {seconds:.1f}s", flush=True)
    return path


def run(plan_path, output):
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    tasks = [tuple(task) for task in plan["tasks"]]
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("Invalid task assignments")
    for horizon, outcome, _ in tasks:
        if imp.cache_stamp(horizon, outcome) != plan["stamps"][f"{outcome}/h{horizon}"]:
            raise ValueError(f"Input or package version mismatch for {outcome}/h{horizon}")
    workers = min(len(tasks), int(os.environ["SLURM_CPUS_PER_TASK"]) // imp.THREADS)
    if workers < 1:
        raise ValueError("The Slurm allocation does not have enough CPUs")
    print(f"{len(tasks)} stages using {workers} processes x {imp.THREADS} CPU threads", flush=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        paths = list(pool.map(compute, tasks))
    temp = output.with_suffix(".tmp")
    with ZipFile(temp, "w", ZIP_DEFLATED, compresslevel=1) as bundle:
        file_hashes = {}
        for path in paths:
            name = path.relative_to(ARTIFACTS).as_posix()
            file_hashes[name] = sha256(path)
            bundle.write(path, name)
        bundle.writestr("receipt.json", json.dumps({
            "tasks": tasks, "sha256": file_hashes, "stamps": plan["stamps"],
            "job_id": os.environ.get("SLURM_JOB_ID"), "host": socket.gethostname(),
            "verified_at_utc": datetime.now(timezone.utc).isoformat()
        }, indent=2))
    temp.replace(output)


def receive(plan_path, archive):
    import numpy as np

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    with ZipFile(archive) as bundle:
        receipt = json.loads(bundle.read("receipt.json"))
        if receipt["tasks"] != plan["tasks"] or receipt["stamps"] != plan["stamps"]:
            raise ValueError("Received archive differs from the assigned task list or input versions")
        validated = []
        import hashlib
        for h, outcome, stage in plan["tasks"]:
            path = imp.cache_path(h, outcome, stage)
            name = path.relative_to(ARTIFACTS).as_posix()
            data = bundle.read(name)
            if hashlib.sha256(data).hexdigest() != receipt["sha256"][name]:
                raise ValueError(f"Invalid checksum: {name}")
            payload = pickle.loads(data)
            dev, ext, predictors = load_cohorts(h, outcome)
            assert payload["stamp"] == imp.cache_stamp(h, outcome)
            assert payload["predictors"] == predictors and payload["imputer_fits"] == 1
            if stage == "external":
                assert_predictors(payload["X_dev"], dev, predictors)
                assert_predictors(payload["X_ext"], ext, predictors)
            else:
                for k, ix in zip(("train", "val", "test"), imp.split_indices(dev.y.to_numpy(), int(stage[3:]))):
                    np.testing.assert_array_equal(payload[k], ix)
                    assert_predictors(payload[f"X_{k}"], dev.iloc[ix], predictors)
            validated.append((path, data))
        for path, data in validated:
            if path.exists():
                print(f"Retaining completed local cache: {path.name}")
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(".unc.tmp")
            temp.write_bytes(data)
            try:
                os.link(temp, path)
                print(f"Imported UNC cache: {path}")
            except FileExistsError:
                print(f"Retaining concurrently completed local cache: {path.name}")
            finally:
                temp.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "receive"))
    parser.add_argument("plan", type=Path)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    (run if args.action == "run" else receive)(args.plan, args.archive)
