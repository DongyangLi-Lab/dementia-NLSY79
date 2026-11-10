"""Paths, atomic writes, and hash-verified stage receipts."""
import json
import os
from pathlib import Path
import pickle
import tempfile

from train.cohorts import ARTIFACTS, sha256


def job_dir(kind, horizon, outcome, model):
    return ARTIFACTS / kind / outcome / f"h{horizon}" / model


def write_file(path, write):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
    try:
        write(temp)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def save_json(path, data):
    write_file(path, lambda p: p.write_text(json.dumps(data, indent=2), encoding="utf-8"))


def save_model(path, model):
    def write(p):
        with p.open("wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    write_file(path, write)


def load_model(path):
    with path.open("rb") as stream:
        return pickle.load(stream)


def save_receipt(path, stamp, files):
    save_json(path, {"stamp": stamp, "sha256": {p.name: sha256(p) for p in files}})


def verified(path, stamp, files):
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        return receipt == {"stamp": stamp, "sha256": {p.name: sha256(p) for p in files}}
    except (OSError, ValueError):
        return False
