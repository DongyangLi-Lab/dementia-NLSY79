import hashlib, os, pickle
from pathlib import Path
import numpy as np
from train import imputation as imp
from train.cohorts import load_cohorts
from train.prepared import assert_predictors
src = Path(r'C:\Users\LDY\Downloads\rep6.pkl')
data = src.read_bytes()
assert hashlib.sha256(data).hexdigest() == '42355cc7102d521c1b7a8b9daa4adeeb7f6436a6187cdbf6bb6c7d85dd892b39'
payload = pickle.loads(data)
dev, ext, predictors = load_cohorts(8, 'any')
assert payload['stamp'] == imp.cache_stamp(8, 'any')
assert payload['predictors'] == predictors and payload['imputer_fits'] == 1
for k, ix in zip(('train','val','test'), imp.split_indices(dev.y.to_numpy(), 6)):
    np.testing.assert_array_equal(payload[k], ix)
    assert_predictors(payload[f'X_{k}'], dev.iloc[ix], predictors)
path = imp.cache_path(8, 'any', 'rep6')
tmp = path.with_suffix('.unc.tmp')
tmp.write_bytes(data)
try:
    os.link(tmp, path)
    print('Validated and imported any/h8/rep6')
except FileExistsError:
    print('Validated remote cache; retained existing local any/h8/rep6')
finally:
    tmp.unlink()
