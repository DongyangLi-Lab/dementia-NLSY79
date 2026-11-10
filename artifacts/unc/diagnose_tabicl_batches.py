import json
import numpy as np
from tabicl.model.inference import InferenceManager
from inference.importance import load_temporal_model
from inference.scoring import predict
from inference import evaluation

seen = set()
original = InferenceManager.estimate_safe_batch_size
def record(self, *args, **kwargs):
    result = original(self, *args, **kwargs)
    record = (self.enc_name, str(args), str(kwargs), round(result[0], 1), result[1])
    if record not in seen:
        seen.add(record)
        print('BATCH_POLICY', record, flush=True)
    return result
InferenceManager.estimate_safe_batch_size = record
m, xd, xe, y, ids, predictors = load_temporal_model('TabICL', 6)
base = predict(m, xe)
saved = evaluation.load(6, 'TabICL', 'temporal')['p'].to_numpy()
print('REPLAY', json.dumps({'max_abs_diff': float(np.max(np.abs(base-saved))), 'allclose': bool(np.allclose(base,saved,rtol=1e-5,atol=1e-7))}), flush=True)
