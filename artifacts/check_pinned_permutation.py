"""Check pinned inference against one preserved permutation without overwriting it."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from artifacts import tabicl_pinned_permutations as pinned
from artifact_store import save_json
from train.cohorts import ARTIFACTS, sha256

i = int(sys.argv[1])
j = int(sys.argv[2]) if len(sys.argv) > 2 else 0
pinned.module.WORK = ARTIFACTS / 'tabicl_pinned_probe'
original = pinned.module.TARGET / 'permutations' / f'{i:02d}.npy'
digest = sha256(original)
pinned.module.worker(i, j)
actual = np.load(pinned.module.WORK / f'{i:02d}-{j}.npy')
expected = np.load(original)[j]
passed = bool(np.allclose(actual, expected, rtol=1e-5, atol=1e-7))
result = {'group': i, 'replicate': j, 'passed': passed,
          'max_abs': float(np.max(np.abs(actual-expected))), 'original_sha256': digest,
          'pinned_sha256': sha256(Path(pinned.__file__)),
          'prediction_sha256': sha256(pinned.module.WORK / f'{i:02d}-{j}.npy')}
assert digest == sha256(original)
save_json(pinned.module.WORK / f'{i:02d}-{j}.check.json', result)
print(result, flush=True)
if not passed:
    raise SystemExit(1)
