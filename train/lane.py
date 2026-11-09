"""Run one device's model assignments using the verified shared data."""
import argparse
import traceback
import warnings

from sklearn.exceptions import ConvergenceWarning

from artifact_store import save_json
from inference.predict import run as infer
from model.specs import MODELS
from train.benchmark import run as train
from train.cohorts import ARTIFACTS
from train.prepared import COMBINATIONS, require_prepared


def run(device, models):
    require_prepared()
    pending = [(outcome, horizon, model) for horizon, outcome in COMBINATIONS for model in models]
    finished, failures = [], {}
    path = ARTIFACTS / "logs" / f"lane-{device}-status.json"
    save_json(path, {"completed": finished, "failed": failures, "pending": pending})
    for outcome, horizon, model in pending[:]:
        key = f"{outcome}/h{horizon}/{model}"
        try:
            train(model, horizon, outcome, device, threads=2)
            infer(model, horizon, outcome)
            finished.append(key)
        except Exception as error:
            traceback.print_exc()
            failures[key] = f"{type(error).__name__}: {error}"
        pending.remove((outcome, horizon, model))
        save_json(path, {"completed": finished, "failed": failures, "pending": pending})
    if failures:
        raise RuntimeError(f"{device}: failed={failures}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument("models", nargs="+", choices=MODELS)
    args = parser.parse_args()
    warnings.filterwarnings("once", category=ConvergenceWarning)
    run(args.device, args.models)
