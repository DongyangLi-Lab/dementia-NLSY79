"""Compare small-query predictions with saved full-query held-out predictions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from artifact_store import job_dir, load_model
from inference.scoring import predict
from train.imputation import cache_stamp, read_cache

outcome, horizon = 'final', 6
folder = job_dir('checkpoints', horizon, outcome, 'TabICL')
cache = read_cache(horizon, outcome, 'rep0', cache_stamp(horizon, outcome))
model = load_model(folder / 'rep0.pkl')
saved = pd.read_csv(job_dir('predictions', horizon, outcome, 'TabICL') / 'test_predictions.csv')
saved = saved[saved['repeat'] == 0]
X = cache['X_test'].iloc[:32]
small = predict(model, X)
reference = saved['p'].to_numpy()[:len(X)]
delta = np.abs(reference - small)
report = {'model': str(folder / 'rep0.pkl'), 'rows': len(X), 'ensemble_batch': model.batch_size,
          'comparison': 'first held-out rows: subset query versus saved full-query predictions',
          'max_absolute_difference': float(delta.max()), 'mean_absolute_difference': float(delta.mean()),
          'allclose_atol_1e-7_rtol_1e-5': bool(np.allclose(reference, small, atol=1e-7, rtol=1e-5))}
Path('artifacts/tabicl_row_probe.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report), flush=True)
if not report['allclose_atol_1e-7_rtol_1e-5']:
    raise SystemExit('Query-row batching probe did not meet tolerance')
