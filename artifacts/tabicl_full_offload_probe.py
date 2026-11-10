"""Replay a full held-out TabICL repeat with offloaded intermediates and audit predictions."""
import json
import sys
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
control = '--control' in sys.argv
safety_only = '--safety-only' in sys.argv
if safety_only:
    for key in ('COL_CONFIG', 'ROW_CONFIG', 'ICL_CONFIG'):
        getattr(model.inference_config_, key).update({'safety_factor': 0.3})
elif not control:
    for key in ('COL_CONFIG', 'ROW_CONFIG', 'ICL_CONFIG'):
        getattr(model.inference_config_, key).update({'offload': True, 'safety_factor': 0.3})
actual = predict(model, cache['X_test'])
reference = saved['p'].to_numpy()
assert len(reference) == len(actual)
delta = np.abs(reference - actual)
report = {'model': str(folder / 'rep0.pkl'), 'rows': len(actual), 'ensemble_batch': model.batch_size,
          'offload': 'original' if (control or safety_only) else True, 'safety_factor': 0.65 if control else 0.3,
          'max_absolute_difference': float(delta.max()), 'mean_absolute_difference': float(delta.mean()),
          'allclose_atol_1e-7_rtol_1e-5': bool(np.allclose(reference, actual, atol=1e-7, rtol=1e-5))}
Path('artifacts/tabicl_full_control_probe.json' if control else ('artifacts/tabicl_full_safety_probe.json' if safety_only else 'artifacts/tabicl_full_offload_probe.json')).write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report), flush=True)
if not report['allclose_atol_1e-7_rtol_1e-5']:
    raise SystemExit('Full-query offload probe did not meet tolerance')
