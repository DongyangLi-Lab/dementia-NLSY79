"""Verify label rules, split coverage, imputation inputs, and prediction receipts."""
import argparse
import json

import numpy as np
import pandas as pd

from artifact_store import job_dir, save_json, verified
from model.specs import MODELS, GPU_MODELS
from train.cohorts import ARTIFACTS, WINDOWS, load_cohorts, sha256
from train.imputation import N_REPEATS, split_indices
from train.prepared import PREPARED_RECEIPT, require_prepared, verify_imputations


def verify_cohorts():
    path = ARTIFACTS / "cohorts"
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    assert all(sha256(path / name) == digest for name, digest in manifest["cohorts_sha256"].items())
    counts = []
    for baseline, final in WINDOWS:
        window = f"{baseline}-{final}"
        ledger = pd.read_csv(path / f"{window}_audit.csv").set_index("hhidpn", verify_integrity=True)
        assessments = ledger.filter(regex=r"^dementia_")
        observed = assessments.eq(1).any(axis=1)
        for outcome in ("final", "any"):
            cohort = pd.read_csv(path / f"{window}_{outcome}.csv").set_index("hhidpn", verify_integrity=True)
            expected = ledger[f"y_{outcome}"].dropna()
            pd.testing.assert_series_equal(cohort.y, expected, check_dtype=False, check_names=False)
            selected = ledger.loc[cohort.index]
            assert not selected.death_by_final_wave.any() and selected.tracker_final_vital.eq(1).all()
            assert observed.loc[cohort.index[cohort.y.eq(1)]].all()
            neg = cohort.index[cohort.y.eq(0)]
            assert not observed.loc[neg].any()
            control = assessments.iloc[:, -1].eq(0) if outcome == "final" else assessments.eq(0).any(axis=1)
            assert control.loc[neg].all()
            assert set(cohort.index) == set(ledger.index[ledger.alive_at_final_wave & (observed | control)])
            counts.append({"window": window, "outcome": outcome, "eligible": len(ledger),
                           "n": len(cohort), "cases": int(cohort.y.sum()),
                           "deaths_excluded": int(ledger.death_by_final_wave.sum())})
    return counts


def verify_results():
    require_prepared()
    n_jobs = 0
    for outcome in ("final", "any"):
        for h in (6, 8):
            dev, ext, _ = load_cohorts(h, outcome)
            for model in MODELS:
                target = job_dir("predictions", h, outcome, model)
                trained = job_dir("checkpoints", h, outcome, model)
                done = target / "inference.done.json"
                train_done = trained / "training.done.json"
                r_stamp = json.loads(done.read_text(encoding="utf-8"))["stamp"]
                assert r_stamp["training"] == sha256(train_done)
                names = ("test_predictions.csv", "temporal_predictions.csv", "metrics.csv")
                assert verified(done, r_stamp, tuple(target / name for name in names))
                test, temporal, metric = [pd.read_csv(target / name) for name in names]
                for r, frame in test.groupby("repeat"):
                    ix = split_indices(dev.y.to_numpy(), r)[2]
                    np.testing.assert_array_equal(frame[["hhidpn", "y"]], dev.iloc[ix][["hhidpn", "y"]])
                assert set(test.repeat) == set(range(N_REPEATS))
                np.testing.assert_array_equal(temporal[["hhidpn", "y"]], ext[["hhidpn", "y"]])
                assert test.p.between(0, 1).all() and temporal.p.between(0, 1).all()
                assert len(metric) == 2 * (N_REPEATS + 1)
                selection = json.loads((trained / "selection.json").read_text(encoding="utf-8"))
                assert (selection["device"]["backend"] != "CPU") == (model in GPU_MODELS)
                n_jobs += 1
    return {"imputations": 4 * (N_REPEATS + 1), "model_runs": n_jobs, "predictors": 60, "model_inputs": 94}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--cohorts-only", action="store_true")
    action.add_argument("--prepared-only", action="store_true")
    args = parser.parse_args()
    result = {"cohorts": verify_cohorts()}
    if args.cohorts_only:
        path = ARTIFACTS / "cohort_verification.json"
    elif args.prepared_only:
        result.update(verify_imputations())
        path = PREPARED_RECEIPT
    else:
        result.update(verify_results())
        path = ARTIFACTS / "verification.json"
    save_json(path, result)
    print(f"Verification passed: {path}", flush=True)
