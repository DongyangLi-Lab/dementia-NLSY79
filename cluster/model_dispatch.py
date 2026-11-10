"""Package the verified inputs and exchange isolated UNC model results."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import os
import socket
from zipfile import ZipFile, ZIP_DEFLATED

from artifact_store import job_dir, save_json, verified, write_file
from model.specs import MODELS
from train.cohorts import ARTIFACTS, ROOT, sha256
from train.imputation import N_REPEATS
from train.prepared import COMBINATIONS, PREPARED_RECEIPT, cache_files, require_prepared


REMOTE = {"benchmark": [("final", 8, "GAT"), ("any", 6, "GAT"), ("any", 8, "GAT")],
          "comparison": [("final", 6, name) for name in ("LR", "GNB", "DT")]}


def input_bundle(output):
    require_prepared()
    sources = [ROOT / "artifact_store.py"]
    for package in ("model", "train", "inference"):
        sources.extend(sorted((ROOT / package).glob("*.py")))
    sources.append(Path(__file__))
    paths = [*sources, PREPARED_RECEIPT, *cache_files().values(),
             *sorted((ARTIFACTS / "cohorts").glob("*.csv")), ARTIFACTS / "cohorts/manifest.json"]
    digests = {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}
    def write(temp):
        with ZipFile(temp, "w", ZIP_DEFLATED, compresslevel=1) as archive:
            for path in paths:
                archive.write(path, path.relative_to(ROOT).as_posix())
            archive.writestr("input_sha256.json", json.dumps(digests, indent=2))
    write_file(output, write)
    print(f"Verified input bundle: {output} | {sha256(output)} | {len(cache_files())} imputations", flush=True)


def check_bundle():
    expected = json.loads((ROOT / "input_sha256.json").read_text(encoding="utf-8"))
    for name, digest in expected.items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"Transferred input differs: {name}")
    require_prepared()
    print("Verified all synchronized source files, cohort files and 44 imputation caches", flush=True)


def artifact_paths(kind, outcome, horizon, model):
    if kind == "comparison":
        folder = job_dir("imputation_comparison", horizon, outcome, model)
        return [folder / f"{strategy}-rep{r}{suffix}" for strategy in ("mean", "complete_case")
                for r in range(N_REPEATS) for suffix in (".csv", "-predictions.csv", ".done.json")]
    checkpoints = job_dir("checkpoints", horizon, outcome, model)
    predictions = job_dir("predictions", horizon, outcome, model)
    return ([checkpoints / name for name in ("selection.json", "training.done.json", "validation_grid.csv", "temporal.pkl")]
            + [checkpoints / f"rep{r}{suffix}" for r in range(N_REPEATS) for suffix in (".json", ".pkl", ".done.json")]
            + [predictions / name for name in ("test_predictions.csv", "temporal_predictions.csv", "metrics.csv", "inference.done.json")])


def run_remote(kind, outcome, horizon, model, output):
    if (outcome, horizon, model) not in REMOTE[kind]:
        raise ValueError("Task is outside the UNC assignment")
    require_prepared()
    if kind == "benchmark":
        from train.benchmark import run as train
        from inference.predict import run as infer
        train(model, horizon, outcome, "0", threads=2)
        infer(model, horizon, outcome)
    else:
        from train.imputation_comparison import run as compare
        compare(model, "cpu")
    paths = artifact_paths(kind, outcome, horizon, model)
    digests = {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}
    receipt = {"kind": kind, "outcome": outcome, "horizon": horizon, "model": model, "site": "UNC",
               "job_id": os.environ.get("SLURM_JOB_ID"), "host": socket.gethostname(),
               "prepared_sha256": sha256(PREPARED_RECEIPT), "files": digests}
    def write(temp):
        with ZipFile(temp, "w", ZIP_DEFLATED, compresslevel=1) as archive:
            for path in paths:
                archive.write(path, path.relative_to(ROOT).as_posix())
            archive.writestr("remote_result.json", json.dumps(receipt, indent=2))
    write_file(output, write)
    print(f"Packaged {kind} {outcome}/h{horizon}/{model}: {sha256(output)}", flush=True)


def receive(archive_path):
    import hashlib
    with ZipFile(archive_path) as archive:
        receipt = json.loads(archive.read("remote_result.json"))
        kind = receipt["kind"]
        task = (receipt["outcome"], receipt["horizon"], receipt["model"])
        if task not in REMOTE[kind] or receipt["prepared_sha256"] != sha256(PREPARED_RECEIPT):
            raise ValueError("Remote result does not match a dispatched task or its prepared inputs")
        paths = artifact_paths(kind, *task)
        names = [path.relative_to(ROOT).as_posix() for path in paths]
        if len(archive.namelist()) != len(names)+1 or set(archive.namelist()) != {*names, "remote_result.json"}:
            raise ValueError("Unexpected files in remote result")
        for name in names:
            with archive.open(name) as stream:
                digest = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != receipt["files"][name]:
                raise ValueError(f"Result checksum failed for {name}")
        for path in paths:
            if path.exists() and sha256(path) != receipt["files"][path.relative_to(ROOT).as_posix()]:
                raise ValueError(f"Conflict with an existing result: {path}")
        # Publish completion markers last, so the local verifier never sees a partial task.
        for path in sorted(paths, key=lambda p: p.name.endswith(".done.json")):
            if not path.exists():
                def copy(temp):
                    import shutil
                    with archive.open(path.relative_to(ROOT).as_posix()) as source, temp.open("wb") as target:
                        shutil.copyfileobj(source, target)
                write_file(path, copy)
    save_json(ARTIFACTS / "unc" / f"received-{kind}-{task[0]}-h{task[1]}-{task[2]}.json", receipt)
    print(f"Imported {kind} {task} from UNC job {receipt['job_id']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    command = parser.add_subparsers(dest="action", required=True)
    command.add_parser("verify-inputs")
    bundle = command.add_parser("bundle")
    bundle.add_argument("archive", type=Path)
    run = command.add_parser("run")
    run.add_argument("kind", choices=REMOTE)
    run.add_argument("outcome", choices=("final", "any"))
    run.add_argument("horizon", type=int, choices=(6, 8))
    run.add_argument("model", choices=MODELS)
    run.add_argument("archive", type=Path)
    inp = command.add_parser("receive")
    inp.add_argument("archive", type=Path)
    args = parser.parse_args()
    if args.action == "bundle":
        input_bundle(args.archive)
    elif args.action == "verify-inputs":
        check_bundle()
    elif args.action == "run":
        run_remote(args.kind, args.outcome, args.horizon, args.model, args.archive)
    else:
        receive(args.archive)
