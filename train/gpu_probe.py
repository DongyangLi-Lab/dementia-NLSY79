"""Confirm that all five GPU-enabled model families can fit, reload and predict."""
import io
import pickle

import numpy as np
import pandas as pd
import torch

from inference.scoring import predict
from model.specs import configuration

if __name__ == "__main__":
    import sys
    device, *models = sys.argv[1:]
    x = pd.DataFrame(np.random.default_rng(42).normal(size=(512, 94)).astype(np.float32))
    y = pd.Series((x[0] > 0.7).astype(int))
    for name in models:
        torch.cuda.reset_peak_memory_stats()
        cls, _, fixed = configuration(name, device, 2)
        small = {"CatBoost": {"iterations": 3}, "RealMLP": {"n_epochs": 2},
                 "GAT": {"max_iter": 1, "max_restarts": 0, "reference_size": 24},
                 "TabICL": {"n_estimators": 2, "batch_size": 2},
                 "LightGBM": {"n_estimators": 3, "verbosity": 1}}
        model = cls(**{**fixed, **small[name]})
        model.fit(x, y)
        expected = predict(model, x.iloc[:24])
        stream = io.BytesIO(pickle.dumps(model))
        actual = predict(pickle.load(stream), x.iloc[:24])
        assert np.allclose(expected, actual, rtol=1e-6, atol=1e-7)
        print(name, torch.cuda.get_device_name(0), "replay PASS",
              "torch peak allocated MiB", torch.cuda.max_memory_allocated() / 2**20, flush=True)
