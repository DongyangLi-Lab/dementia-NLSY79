"""Shared preprocessing checks and receipt required by the model workers."""
import json

import numpy as np

from train.cohorts import ARTIFACTS, NON_PREDICTORS, load_cohorts, sha256
from train.imputation import N_REPEATS, cache_path, cache_stamp, category_groups, read_cache, split_indices

COMBINATIONS = ((6, "final"), (8, "final"), (6, "any"), (8, "any"))
PREPARED_RECEIPT = ARTIFACTS / "prepared_inputs.json"


def cache_files():
    return {
        f"{outcome}/h{h}/{name}": cache_path(h, outcome, name)
        for h, outcome in COMBINATIONS
        for name in (*(f"rep{r}" for r in range(N_REPEATS)), "external")
    }


def assert_predictors(X, cohort, predictors):
    assert list(X) == predictors and len(X) == len(cohort)
    observed = cohort[predictors].notna().to_numpy()
    assert np.isfinite(X.to_numpy()).all()
    assert np.allclose(X.to_numpy()[observed], cohort[predictors].to_numpy()[observed])


def verify_imputations():
    stamps = {}
    digests = {}
    for h, outcome in COMBINATIONS:
        dev, ext, predictors = load_cohorts(h, outcome)
        assert len(set(predictors)) == 94 and not set(predictors).intersection(NON_PREDICTORS)
        assert len(predictors) - sum(len(x) - 1 for x in category_groups(predictors).values()) == 60
        stamp = cache_stamp(h, outcome)
        prefix = f"{outcome}/h{h}"
        stamps[prefix] = stamp
        for r in range(N_REPEATS):
            name = f"rep{r}"
            c = read_cache(h, outcome, name, stamp)
            assert c["imputer_fits"] == 1 and c["predictors"] == predictors
            for key, ix in zip(("train", "val", "test"), split_indices(dev.y.to_numpy(), r)):
                np.testing.assert_array_equal(c[key], ix)
                assert_predictors(c[f"X_{key}"], dev.iloc[ix], predictors)
            digests[f"{prefix}/{name}"] = sha256(cache_path(h, outcome, name))
        c = read_cache(h, outcome, "external", stamp)
        assert c["imputer_fits"] == 1 and c["predictors"] == predictors
        assert_predictors(c["X_dev"], dev, predictors)
        assert_predictors(c["X_ext"], ext, predictors)
        digests[f"{prefix}/external"] = sha256(cache_path(h, outcome, "external"))
    return {"imputations": len(digests), "predictors": 60, "model_inputs": 94,
            "cache_stamps": stamps, "cache_sha256": digests}


def require_prepared():
    try:
        receipt = json.loads(PREPARED_RECEIPT.read_text(encoding="utf-8"))
        files = cache_files()
        stamps = {f"{outcome}/h{h}": cache_stamp(h, outcome) for h, outcome in COMBINATIONS}
        if (receipt["cache_stamps"] != stamps or receipt["imputations"] != len(files)
                or receipt["cache_sha256"] != {name: sha256(path) for name, path in files.items()}):
            raise ValueError("The prepared inputs changed after verification")
    except (OSError, ValueError, KeyError) as error:
        raise RuntimeError("All four sets of imputations must pass preprocessing verification before training") from error
