"""Replay frozen checkpoints on held-out participants and save predictions and metrics."""
import argparse
import json
import time

import numpy as np
import pandas as pd

from artifact_store import job_dir, load_model, save_receipt, verified
from inference.scoring import metrics, predict
from model.specs import MODELS
from train.cohorts import ROOT, atomic_csv, load_cohorts, sha256
from train.imputation import cache_stamp, read_cache


def run(model_name, horizon, outcome):
    trained = job_dir("checkpoints", horizon, outcome, model_name)
    train_done = trained / "training.done.json"
    train_stamp = json.loads(train_done.read_text(encoding="utf-8"))["stamp"]
    training_files = [trained / n for n in ("selection.json", "validation_grid.csv", "temporal.pkl")]
    stamp = {"training": sha256(train_done), "inference": sha256(ROOT / "inference/predict.py")}
    current_imputation = cache_stamp(horizon, outcome)
    if train_stamp["imputation"] != current_imputation or not verified(train_done, train_stamp, training_files):
        raise ValueError(f"Invalid training receipt: {trained}")
    selection = json.loads(training_files[0].read_text(encoding="utf-8"))
    for rep in selection["per_repeat"]:
        key = f"rep{rep['repeat']}"
        receipt = trained / f"{key}.done.json"
        if not verified(receipt, train_stamp, (trained / f"{key}.json", trained / f"{key}.pkl")):
            raise ValueError(f"Invalid repeat receipt: {receipt}")
        stamp[key] = sha256(receipt)
    target = job_dir("predictions", horizon, outcome, model_name)
    files = [target / name for name in ("test_predictions.csv", "temporal_predictions.csv", "metrics.csv")]
    done = target / "inference.done.json"
    if verified(done, stamp, files):
        return

    dev, ext, predictors = load_cohorts(horizon, outcome)
    rows, internal = [], []

    def score(model_file, X, cohort, setting, selected):
        model = load_model(model_file)
        start = time.perf_counter()
        p = predict(model, X)
        seconds = time.perf_counter() - start
        del model
        y = cohort.y.to_numpy()
        complete = cohort[predictors].notna().all(axis=1).to_numpy()
        repeat = selected.get("repeat", -1)
        for population, mask in (("all", np.ones(len(y), bool)), ("complete", complete)):
            rows.append({"setting": setting, "population": population, "repeat": repeat,
                         "config": selected["config"], **metrics(y[mask], p[mask], selected["threshold"]),
                         "fit_seconds": selected["fit_seconds"], "prediction_seconds": seconds,
                         "restarts": selected["restarts"], "collapsed": selected["collapsed"]})
        result = pd.DataFrame({"hhidpn": cohort.hhidpn.to_numpy(), "y": y, "p": p, "complete": complete})
        if repeat >= 0:
            result.insert(0, "repeat", repeat)
        return result

    for rep in selection["per_repeat"]:
        name = f"rep{rep['repeat']}"
        c = read_cache(horizon, outcome, name, current_imputation)
        internal.append(score(trained / f"{name}.pkl", c["X_test"], dev.iloc[c["test"]], "internal", rep))
    c = read_cache(horizon, outcome, "external", current_imputation)
    temporal = score(trained / "temporal.pkl", c["X_ext"], ext, "temporal", selection["temporal"])
    target.mkdir(parents=True, exist_ok=True)
    for frame, path in zip((pd.concat(internal, ignore_index=True), temporal, pd.DataFrame(rows)), files):
        atomic_csv(frame, path)
    save_receipt(done, stamp, files)
    print(f"{outcome}/h{horizon}/{model_name} held-out inference complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--horizon", type=int, choices=(6, 8), required=True)
    parser.add_argument("--outcome", choices=("final", "any"), required=True)
    args = parser.parse_args()
    run(args.model, args.horizon, args.outcome)
