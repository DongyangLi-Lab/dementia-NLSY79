"""Compare saved TabICL predictions across ensemble batch sizes on real held-out rows."""
import json
from pathlib import Path
import numpy as np
from artifact_store import job_dir, load_model
from inference.scoring import predict
from train.imputation import cache_stamp, read_cache

outcome, horizon = 'final', 6
folder = job_dir('checkpoints', horizon, outcome, 'TabICL')
cache = read_cache(horizon, outcome, 'rep0', cache_stamp(horizon, outcome))
model = load_model(folder / 'rep0.pkl')
X = cache['X_test'].iloc[:32]
original_batch = model.batch_size
baseline = predict(model, X)
baseline_repeat = predict(model, X)
model.batch_size = 1
smaller = predict(model, X)
model.batch_size = original_batch
baseline_after = predict(model, X)
delta = np.abs(baseline - smaller)
report = {'model': str(folder / 'rep0.pkl'), 'rows': len(X), 'original_ensemble_batch': original_batch,
          'reduced_ensemble_batch': 1, 'max_absolute_difference': float(delta.max()),
          'mean_absolute_difference': float(delta.mean()),
          'same_batch_repeat_max_difference': float(np.max(np.abs(baseline - baseline_repeat))),
          'same_batch_after_max_difference': float(np.max(np.abs(baseline - baseline_after))),
          'allclose_atol_1e-7_rtol_1e-5': bool(np.allclose(baseline, smaller, atol=1e-7, rtol=1e-5))}
Path('artifacts/tabicl_batch_probe.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report), flush=True)
if not report['allclose_atol_1e-7_rtol_1e-5']:
    raise SystemExit('Batch-size probe did not meet tolerance')
