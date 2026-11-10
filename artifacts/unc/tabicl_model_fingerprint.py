import hashlib, json, sys
import numpy as np
from inference import importance as imp
from pathlib import Path

if sys.platform != 'win32':
    import pickle, pathlib
    class Compatible(pickle.Unpickler):
        def find_class(self, module, name):
            return pathlib.PureWindowsPath if (module, name) == ('pathlib', 'WindowsPath') else super().find_class(module, name)
    def load(p):
        with open(p, 'rb') as f:
            return Compatible(f).load()
    imp.load_model = load
m, _, xe, _, _, _ = imp.load_temporal_model('TabICL', 6)
def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
h = hashlib.sha256()
for name, t in m.model_.state_dict().items():
    h.update(name.encode()); h.update(t.detach().cpu().contiguous().numpy().tobytes())
print('MODEL_STATE_SHA256', h.hexdigest(), flush=True)
for name, pp in m.ensemble_generator_.preprocessors_.items():
    print('PREPROCESSOR', name, 'SCALER_MEAN', digest(pp.standard_scaler_.mean_),
          'SCALE', digest(pp.standard_scaler_.scale_),
          'LOWER', digest(pp.outlier_remover_.lower_bounds_),
          'UPPER', digest(pp.outlier_remover_.upper_bounds_), flush=True)
    if pp.normalizer_ is not None:
        print('NORMALIZER', type(pp.normalizer_).__name__, 'LAMBDA', digest(pp.normalizer_.lambdas_),
              'MEAN', digest(pp.normalizer_._scaler.mean_), 'SCALE', digest(pp.normalizer_._scaler.scale_), flush=True)
print('ENSEMBLE_CONFIG', hashlib.sha256(repr(m.ensemble_generator_.ensemble_configs_).encode()).hexdigest(), flush=True)
