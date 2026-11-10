import hashlib
import importlib.metadata as md
import numpy as np
from inference.importance import load_temporal_model

print('VERSIONS', {x: md.version(x) for x in ['scipy', 'pandas', 'joblib', 'torch', 'scikit-learn', 'numpy', 'tabicl']}, flush=True)
model, xd, xe, y, ids, predictors = load_temporal_model('TabICL', 6)
def digest(a):
    a = np.ascontiguousarray(a)
    return str(a.shape), str(a.dtype), hashlib.sha256(a.tobytes()).hexdigest()
print('XE', digest(xe.to_numpy()), flush=True)
encoded = model.X_encoder_.transform(xe)
print('ENCODED', digest(encoded), flush=True)
f = model.ensemble_generator_.unique_filter_.transform(encoded)
print('FILTERED', digest(f), flush=True)
for key, pp in model.ensemble_generator_.preprocessors_.items():
    print('TRAIN', key, digest(pp.X_transformed_), flush=True)
    scaled = pp.standard_scaler_.transform(f)
    print('SCALED', key, digest(scaled), flush=True)
    normalized = pp.normalizer_.transform(scaled) if pp.normalizer_ is not None else scaled
    print('NORMALIZED', key, digest(normalized), flush=True)
    print('TRANSFORM', key, digest(pp.transform(f)), flush=True)
for key, (x, labels) in model.ensemble_generator_.transform(encoded).items():
    print('ENSEMBLE', key, digest(x), digest(labels), flush=True)
