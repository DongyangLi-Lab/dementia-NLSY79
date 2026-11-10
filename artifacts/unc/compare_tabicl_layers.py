import json
import numpy as np
from inference.importance import load_temporal_model

model, _, xe, _, _, _ = load_temporal_model('TabICL', 6)
f = model.ensemble_generator_.unique_filter_.transform(model.X_encoder_.transform(xe))
remote = np.load(r'C:\Users\LDY\Downloads\remote_layers_diagnostic.npz')
report = {}
for key, pp in model.ensemble_generator_.preprocessors_.items():
    scaled = pp.standard_scaler_.transform(f)
    local = {key + '_scaled': scaled, key + '_transformed': pp.transform(f)}
    if pp.normalizer_ is not None:
        local[key + '_normalized'] = pp.normalizer_.transform(scaled)
    for name, a in local.items():
        b = remote[name]
        delta = np.abs(a.astype('float64') - b.astype('float64'))
        report[name] = {'shape': list(a.shape), 'max': float(delta.max()), 'mean': float(delta.mean()),
                        'p99': float(np.quantile(delta, .99)), 'different': int((delta != 0).sum())}
print(json.dumps(report, indent=2))
