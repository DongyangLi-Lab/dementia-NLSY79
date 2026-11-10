"""Tune on development validation only and save fitted models for separate inference."""
import argparse
import importlib.metadata as metadata
import itertools
import json
import os
from pathlib import Path
import platform
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score

from artifact_store import job_dir, save_json, save_model, save_receipt, verified
from inference.scoring import best_f1_threshold, predict
from model import specs
from train.cohorts import ROOT, load_cohorts, sha256
from train.imputation import N_REPEATS, cache_path, cache_stamp, read_cache


def fit(cls, params, X, y):
    start = time.perf_counter()
    model = cls(**params)
    model.fit(X.astype(np.float32), pd.Series(y, dtype=int))
    return model, time.perf_counter() - start


def device_metadata(device, model_name):
    if device == "cpu":
        return {"backend": "CPU"}
    import torch
    gpu_name = torch.cuda.get_device_name(0)
    torch.ones(1, device="cuda:0").cpu()
    free, total = torch.cuda.mem_get_info(0)
    return {"backend": "OpenCL" if model_name == "LightGBM" else "CUDA",
            "gpu_index": int(device), "gpu_name": gpu_name, "gpu_uuid": os.environ["CUDA_VISIBLE_DEVICES"],
            "opencl_device": os.environ.get("HRS_OPENCL_DEVICE") if model_name == "LightGBM" else None,
            "free_vram_bytes_at_start": free, "total_vram_bytes": total}


def model_stamp(horizon, outcome, device, model, fixed, grid):
    versions = {pkg: metadata.version(pkg) for pkg in (
        "numpy", "pandas", "scikit-learn", "torch", "torch-geometric", "lightgbm", "catboost",
        "miceforest", "imbalanced-learn", "pytabkit", "tabicl")}
    versions["python"] = platform.python_version()
    sources = [Path(__file__), ROOT / "model/specs.py", ROOT / "inference/scoring.py"]
    if model == "GAT":
        sources.extend([ROOT / "model/neuralclassifier_fixed.py", ROOT / "model/nnblocks.py"])
    return {"imputation": cache_stamp(horizon, outcome), "device": device, "versions": versions,
            "sources": {p.name: sha256(p) for p in sources}, "fixed": fixed, "grid": grid,
            "repeats": N_REPEATS}


def wait_for_cache(horizon, outcome, stage, stamp):
    path = cache_path(horizon, outcome, stage)
    deadline = time.monotonic() + float(os.environ.get("HRS_CACHE_WAIT_SECONDS", "0"))
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(5)
    return read_cache(horizon, outcome, stage, stamp)


def run(model_name, horizon, outcome, device, threads, available_only=False):
    dev, _, _ = load_cohorts(horizon, outcome)
    y = dev.y.to_numpy()
    cls, grid, fixed = specs.configuration(model_name, device, threads)
    if model_name == "TabICL":
        checkpoint = specs._tabicl_checkpoint(fixed["checkpoint_version"])
        if not checkpoint["sha256"]:
            raise FileNotFoundError("TabICL v1.1 checkpoint must be cached before starting the benchmark")
    else:
        checkpoint = None
    configs = [dict(zip(grid, values)) for values in itertools.product(*grid.values())]
    stamp = model_stamp(horizon, outcome, device, model_name, fixed, grid)
    stamp["checkpoint"] = checkpoint
    stamp["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    device_info = device_metadata(device, model_name)
    target = job_dir("checkpoints", horizon, outcome, model_name)
    selected, all_grids, thresholds = [], [], []
    for r in range(N_REPEATS):
        state_file = target / f"rep{r}.json"
        model_file = target / f"rep{r}.pkl"
        receipt = target / f"rep{r}.done.json"
        if not verified(receipt, stamp, (state_file, model_file)):
            if available_only and not cache_path(horizon, outcome, f"rep{r}").exists():
                return False
            c = wait_for_cache(horizon, outcome, f"rep{r}", stamp["imputation"])
            tr, va = c["train"], c["val"]
            val_predictions, grid_rows = [], []
            for k, cfg in enumerate(configs):
                candidate, fit_seconds = fit(cls, {**fixed, **cfg}, c["X_train"], y[tr])
                start = time.perf_counter()
                p_val = predict(candidate, c["X_val"])
                prediction_seconds = time.perf_counter() - start
                score = float(average_precision_score(y[va], p_val))
                val_predictions.append(p_val)
                grid_rows.append({"repeat": r, "config": k, **{f"param_{n}": str(v) for n, v in cfg.items()},
                                  "val_AUPRC": score, "fit_seconds": fit_seconds,
                                  "prediction_seconds": prediction_seconds,
                                  "restarts": getattr(candidate, "n_restarts_", 0),
                                  "collapsed": getattr(candidate, "collapsed_", False)})
                del candidate
            best = int(np.argmax([row["val_AUPRC"] for row in grid_rows]))
            X_trainval = pd.concat([c["X_train"], c["X_val"]], ignore_index=True)
            model, seconds = fit(cls, {**fixed, **configs[best]}, X_trainval, y[np.r_[tr, va]])
            f1_thresholds = [best_f1_threshold(y[va], p) for p in val_predictions]
            state = {"selection": {"repeat": r, "config": best, "params": configs[best],
                                   "val_AUPRC": grid_rows[best]["val_AUPRC"], "threshold": f1_thresholds[best],
                                   "fit_seconds": seconds, "restarts": getattr(model, "n_restarts_", 0),
                                   "collapsed": getattr(model, "collapsed_", False)},
                     "grid": grid_rows, "thresholds": f1_thresholds, "device": device_info}
            save_model(model_file, model)
            del model
            save_json(state_file, state)
            save_receipt(receipt, stamp, (state_file, model_file))
            print(f"{outcome}/h{horizon}/{model_name}/rep{r} trained", flush=True)
        state = json.loads(state_file.read_text(encoding="utf-8"))
        selected.append(state["selection"])
        all_grids.extend(state["grid"])
        thresholds.append(state["thresholds"])

    grid_df = pd.DataFrame(all_grids)
    k = int(grid_df.groupby("config")["val_AUPRC"].mean().idxmax())
    files = [target / n for n in ("selection.json", "validation_grid.csv", "temporal.pkl")]
    done = target / "training.done.json"
    if verified(done, stamp, files):
        return True
    if available_only and not cache_path(horizon, outcome, "external").exists():
        return False
    cache = wait_for_cache(horizon, outcome, "external", stamp["imputation"])
    model, seconds = fit(cls, {**fixed, **configs[k]}, cache["X_dev"], y)
    save_model(files[2], model)
    grid_df.to_csv(files[1], index=False)
    save_json(files[0], {"model": model_name, "horizon": horizon, "outcome": outcome,
                         "fixed_params": fixed, "grid": grid, "per_repeat": selected, "device": device_info,
                         "temporal": {"config": k, "params": configs[k], "threshold": float(np.median(np.array(thresholds)[:, k])),
                                      "fit_seconds": seconds, "restarts": getattr(model, "n_restarts_", 0),
                                      "collapsed": getattr(model, "collapsed_", False)},
                         "mice_iterations": stamp["imputation"]["iterations"], "versions": stamp["versions"],
                         "checkpoint": checkpoint})
    save_receipt(done, stamp, files)
    print(f"{outcome}/h{horizon}/{model_name} training complete", flush=True)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=specs.MODELS, required=True)
    parser.add_argument("--horizon", type=int, choices=(6, 8), required=True)
    parser.add_argument("--outcome", choices=("final", "any"), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    warnings.filterwarnings("once", category=ConvergenceWarning)
    run(args.model, args.horizon, args.outcome, args.device, args.threads)
