"""Rebuild the held-out six-year comparison of MICE, mean, and complete-case training."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from artifact_store import job_dir, save_receipt, verified
from inference.scoring import best_f1_threshold, metrics, predict
from model.specs import MODELS, configuration
from train.benchmark import device_metadata, fit
from train.cohorts import ARTIFACTS, atomic_csv, load_cohorts, sha256
from train.imputation import N_REPEATS, cache_stamp, read_cache


def run(model_name, device):
    dev, _, inputs = load_cohorts(6)
    cls, grid, fixed = configuration(model_name, device, 2)
    configs = [dict(zip(grid, values)) for values in itertools.product(*grid.values())]
    target = job_dir("imputation_comparison", 6, "final", model_name)
    target.mkdir(parents=True, exist_ok=True)
    device_info = device_metadata(device, model_name)
    stamp = {"imputation": cache_stamp(6, "final"), "implementation": sha256(Path(__file__)),
             "device": {k: v for k, v in device_info.items() if "vram" not in k}}
    X = dev[inputs]
    y = dev.y.to_numpy()
    complete = X.notna().all(axis=1).to_numpy()
    for r in range(N_REPEATS):
        c = read_cache(6, "final", f"rep{r}", stamp["imputation"])
        for strategy in ("mean", "complete_case"):
            result = target / f"{strategy}-rep{r}.csv"
            raw = target / f"{strategy}-rep{r}-predictions.csv"
            receipt = target / f"{strategy}-rep{r}.done.json"
            if verified(receipt, stamp, (result, raw)):
                continue
            tr, va, te = (c[k] for k in ("train", "val", "test"))
            if strategy == "mean":
                mean = X.iloc[tr].mean()
                xtr, xva, xte = (X.iloc[ix].fillna(mean) for ix in (tr, va, te))
            else:
                tr, va, te = (ix[complete[ix]] for ix in (tr, va, te))
                xtr, xva, xte = (X.iloc[ix] for ix in (tr, va, te))
            best_score = -np.inf
            for config in configs:
                estimator, _ = fit(cls, {**fixed, **config}, xtr, y[tr])
                p = predict(estimator, xva)
                ap = average_precision_score(y[va], p)
                del estimator
                if ap > best_score:
                    best_score, best, threshold = ap, config, best_f1_threshold(y[va], p)
            estimator, seconds = fit(cls, {**fixed, **best}, pd.concat([xtr, xva], ignore_index=True), y[np.r_[tr, va]])
            p = predict(estimator, xte)
            del estimator
            atomic_csv(pd.DataFrame([{"strategy": strategy, "model": model_name, "horizon": 6, "repeat": r,
                                      **metrics(y[te], p, threshold), "val_AUPRC": best_score,
                                      "parameters": json.dumps(best), "fit_seconds": seconds,
                                      "backend": device_info["backend"], "gpu": device_info.get("gpu_name", "")}]), result)
            atomic_csv(pd.DataFrame({"hhidpn": dev.hhidpn.to_numpy()[te], "y": y[te], "p": p}), raw)
            save_receipt(receipt, stamp, (result, raw))
            print(f"{model_name} {strategy} repeat {r} complete", flush=True)


def merge():
    from inference.evaluation import load
    from inference.evaluation import corrected_t_interval
    rows = []
    for model in MODELS:
        results = load(6, model, "metrics")
        rows.append(results[(results.setting == "internal") & (results.population == "all")].assign(
            model=model, strategy="miceforest", horizon=6))
        target = job_dir("imputation_comparison", 6, "final", model)
        for strategy in ("mean", "complete_case"):
            rows.extend(pd.read_csv(target / f"{strategy}-rep{r}.csv") for r in range(N_REPEATS))
    repeats = pd.concat(rows, ignore_index=True)
    assert len(repeats) == N_REPEATS * len(MODELS) * 3
    summary = []
    for (model, strategy), frame in repeats.groupby(["model", "strategy"]):
        for metric in ("AUROC", "AUPRC", "Accuracy"):
            est, lo, hi = corrected_t_interval(frame[metric])
            summary.append({"model": model, "strategy": strategy, "metric": metric,
                            "estimate": est, "ci_low": lo, "ci_high": hi,
                            "n_mean": frame.n.mean(), "case_fraction": frame.cases.sum() / frame.n.sum()})
    path = ARTIFACTS / "tables"
    atomic_csv(repeats, path / "imputation_comparison_repeats_h6.csv")
    atomic_csv(pd.DataFrame(summary), path / "S12_imputation_comparison_h6.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--models", nargs="+", choices=MODELS)
    action.add_argument("--merge", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.merge:
        merge()
    else:
        for name in args.models:
            run(name, args.device)
