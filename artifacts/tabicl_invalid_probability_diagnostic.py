"""Inspect the failed group without publishing or repairing its predictions."""
import json
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from artifacts import tabicl_pinned_permutations as pinned
from artifacts.audit_tabicl_permutations import competitors

parser = argparse.ArgumentParser()
parser.add_argument('--label', choices=('first', 'repeat'), default='first')
args = parser.parse_args()
if competitors():
    raise RuntimeError('Diagnostic requires an exclusive GPU window')
model, x, groups = pinned.module.inputs()
index, repeat = 37, 3
name, columns = list(groups.items())[index]
rng = np.random.default_rng(np.random.SeedSequence([2026, index]))
for _ in range(repeat + 1):
    order = rng.permutation(len(x))
xp = x.copy()
xp[columns] = x[columns].to_numpy()[order]
print('INPUT', name, columns, xp.shape, 'finite', bool(np.isfinite(xp.to_numpy()).all()), flush=True)
p = np.asarray(model.predict_proba(xp.astype(np.float32)))
bad = ~np.isfinite(p) | (p < 0) | (p > 1)
out = Path('artifacts/tabicl_invalid_probability_diagnostic')
out.mkdir(exist_ok=True)
path = out / ('37-3-raw.npy' if args.label == 'first' else '37-3-repeat.npy')
np.save(path, p)
report = {'shape': list(p.shape), 'dtype': str(p.dtype), 'nan': int(np.isnan(p).sum()),
          'inf': int(np.isinf(p).sum()), 'out_of_range': int(((p < 0) | (p > 1)).sum()),
          'invalid_rows': np.flatnonzero(bad.any(axis=1)).tolist(),
          'finite_min': float(p[np.isfinite(p)].min()) if np.isfinite(p).any() else None,
          'finite_max': float(p[np.isfinite(p)].max()) if np.isfinite(p).any() else None}
if args.label == 'repeat':
    original = np.load(out / '37-3-raw.npy')
    report['max_abs_vs_first'] = float(np.max(np.abs(p-original)))
    report['matches_first'] = bool(np.allclose(p, original, rtol=1e-5, atol=1e-7))
(out / f'report_{args.label}.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report), flush=True)
