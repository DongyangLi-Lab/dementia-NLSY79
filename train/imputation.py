"""One predictor-only miceforest fit per development split and temporal analysis."""
import argparse
import importlib.metadata
import json
import os
import pickle
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from train.cohorts import ARTIFACTS, NON_PREDICTORS, load_cohorts, sha256

N_REPEATS = 10
MICE_ITERATIONS = 20
THREADS = int(os.environ.get("HRS_THREADS", "4"))

def split_indices(y, repeat):
    idx = np.arange(len(y))
    trval, test = train_test_split(idx, test_size=0.2, stratify=y, random_state=repeat)
    train, val = train_test_split(trval, test_size=0.25, stratify=y[trval], random_state=repeat)
    return np.sort(train), np.sort(val), np.sort(test)


NOMINAL_PREFIXES = ('edstg', 'cendiv', 'mstat', 'raceeth', 'lbrf')


def category_groups(columns):
    """Dummy columns that encode one categorical variable, so they are imputed jointly.

    Nominal variables (reference category = all dummies 0) and each functional item with its "don't do" dummy
    (no difficulty / difficulty / don't do) are each imputed as a single categorical variable, which guarantees
    exactly one category per imputed participant.
    """
    groups = {pre: [c for c in columns if c.startswith(pre)] for pre in NOMINAL_PREFIXES}
    for c in columns:
        if c.endswith('dontdo'):
            groups[f'{c[:-len("dontdo")]}_item'] = [c[:-len('dontdo')], c]
    return {name: cols for name, cols in groups.items() if cols}


def _collapse(X, groups):
    X = X.copy()
    for name, cols in groups.items():
        D = X[cols].to_numpy(dtype=float)
        assert (np.nansum(D, axis=1) <= 1).all(), f'{name}: more than one category set'
        code = np.where(D.sum(axis=1) == 0, 0, np.argmax(D, axis=1) + 1).astype(float)
        code[np.isnan(D).any(axis=1)] = np.nan
        X = X.drop(columns=cols)
        X[name] = pd.Categorical(code, categories=[float(k) for k in range(len(cols) + 1)])
    return X


def _expand(X, groups, columns):
    X = X.copy()
    for name, cols in groups.items():
        code = X.pop(name).astype(float).to_numpy()
        for j, c in enumerate(cols):
            X[c] = (code == j + 1).astype(float)
    return X[columns]


def _impute(fit_X, apply_to, seed):
    """Fit miceforest on fit_X only; return (imputed fit_X, [imputed frames for apply_to]).

    Every predictor gets an imputation model (variable_schema = all columns), so a predictor that is complete in
    fit_X but missing in a later data set is still imputed. Dummy groups are imputed as one categorical variable.
    """
    import miceforest as mf
    if list(fit_X.columns.intersection(NON_PREDICTORS)) or fit_X.shape[1] != 94:
        raise ValueError("Imputation requires the 94 predictor inputs only")
    if any(not X.columns.equals(fit_X.columns) for X in apply_to):
        raise ValueError("Imputer inputs must share the training predictor schema")
    columns = list(fit_X.columns)
    groups = category_groups(columns)
    fit_c = _collapse(fit_X.reset_index(drop=True), groups)
    # Categorical variables use 'fast' mean matching (draw from the predicted class probabilities): rare levels
    # (e.g. "don't do" for picking up a dime) give probabilities of exactly 0 or 1, which break the default
    # log-odds nearest-neighbor matching.
    kernel = mf.ImputationKernel(fit_c, variable_schema=list(fit_c.columns),
                                 mean_match_strategy={name: 'fast' for name in groups}, random_state=seed)
    kernel.mice(MICE_ITERATIONS, num_threads=THREADS)
    fitted = _expand(kernel.complete_data(0), groups, columns)
    applied = [_expand(kernel.impute_new_data(_collapse(X.reset_index(drop=True), groups), random_state=seed,
                                                    iterations=MICE_ITERATIONS)
                       .complete_data(0), groups, columns) for X in apply_to]
    for out, src in zip([fitted] + applied, [fit_X] + list(apply_to)):
        assert not out.isna().any().any(), 'missing values left after imputation'
        observed = src.reset_index(drop=True).notna().to_numpy()
        assert np.allclose(out.to_numpy()[observed], src.to_numpy(dtype=float)[observed]), 'observed values changed'
    return fitted, applied



def cache_path(horizon, outcome, name):
    return ARTIFACTS / "imputation" / outcome / f"h{horizon}" / f"{name}.pkl"


def cache_stamp(horizon, outcome):
    return {
        "protocol": "training-only-v1",
        "iterations": MICE_ITERATIONS,
        "versions": {p: importlib.metadata.version(p) for p in ("miceforest", "lightgbm", "scikit-learn", "pandas")},
        "cohort_sha256": [sha256(ARTIFACTS / "cohorts" / f"{y}-{y+horizon}_{outcome}.csv")
                          for y in (2000, 2010)],
        "implementation": sha256(Path(__file__)),
    }


def read_cache(horizon, outcome, name, stamp):
    with cache_path(horizon, outcome, name).open("rb") as stream:
        payload = pickle.load(stream)
    if payload["stamp"] != stamp:
        raise ValueError(f"Outdated imputation: {outcome}/h{horizon}/{name}")
    return payload


def build(horizon, outcome, repeats=range(N_REPEATS)):
    dev, ext, predictors = load_cohorts(horizon, outcome)
    x_dev = dev[predictors].astype(float)
    y = dev.y.to_numpy()
    stamp = cache_stamp(horizon, outcome)
    stages = [f"rep{r}" for r in repeats] + ["external"]
    for stage in stages:
        path = cache_path(horizon, outcome, stage)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            try:
                read_cache(horizon, outcome, stage, stamp)
                print(f"Reusing {outcome}/h{horizon}/{stage}", flush=True)
                continue
            except (ValueError, EOFError, pickle.UnpicklingError):
                pass
        started = time.perf_counter()
        if stage == "external":
            fit, (external,) = _impute(x_dev, [ext[predictors].astype(float)], seed=999)
            payload = {"X_dev": fit, "X_ext": external}
        else:
            r = int(stage[3:])
            tr, va, te = split_indices(y, r)
            fit, (val, test) = _impute(x_dev.iloc[tr], [x_dev.iloc[va], x_dev.iloc[te]], seed=42+r)
            payload = {"train": tr, "val": va, "test": te, "X_train": fit, "X_val": val, "X_test": test}
        seconds = time.perf_counter() - started
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as stream:
            pickle.dump({**payload, "stamp": stamp, "predictors": predictors, "imputer_fits": 1,
                         "seconds": seconds, "device": "CPU"}, stream, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
        print(f"{outcome}/h{horizon}/{stage} complete in {seconds:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, required=True, choices=(6, 8))
    parser.add_argument("--outcome", choices=("final", "any"), required=True)
    args = parser.parse_args()
    build(args.horizon, args.outcome)
