import numpy as np
from inference.importance import load_temporal_model
m, _, xe, _, _, _ = load_temporal_model('TabICL', 6)
f = m.ensemble_generator_.unique_filter_.transform(m.X_encoder_.transform(xe))
out = {k: p.transform(f) for k, p in m.ensemble_generator_.preprocessors_.items()}
np.savez_compressed('artifacts/unc/local_transform_diagnostic.npz', **out)
