"""Compare original TabICL inference with CPU-offloaded intermediates on real rows."""
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
original = predict(model, X)
repeat = predict(model, X)
config = model.inference_config_
for key in ('COL_CONFIG', 'ROW_CONFIG', 'ICL_CONFIG'):
    getattr(config, key).update({'offload': True, 'safety_factor': 0.3})
offloaded = predict(model, X)
delta = np.abs(original - offloaded)
report = {'model': str(folder / 'rep0.pkl'), 'rows': len(X), 'ensemble_batch': model.batch_size,
          'offload': True, 'safety_factor': 0.3,
          'max_absolute_difference': float(delta.max()), 'mean_absolute_difference': float(delta.mean()),
          'same_setting_repeat_max_difference': float(np.max(np.abs(original - repeat))),
          'allclose_atol_1e-7_rtol_1e-5': bool(np.allclose(original, offloaded, atol=1e-7, rtol=1e-5))}
Path('artifacts/tabicl_offload_probe.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report), flush=True)
if not report['allclose_atol_1e-7_rtol_1e-5']:
    raise SystemExit('Offload probe did not meet tolerance')
