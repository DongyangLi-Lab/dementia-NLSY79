"""Link successful diagnostic stages to the real sequential/isolated SHAP control."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from artifact_store import save_json
from artifacts import tabicl_kernel_shap_sharded as shap
from train.cohorts import ARTIFACTS, sha256

root = ARTIFACTS / 'unc' / 'projection_fixtures_private'
full_short = root / '20261110_002014_320060'
coalition = root / '20261110_002014_320060_coalition_retry' / 'coalition'
if not shap.check_control():
    raise ValueError('The actual sequential/isolated GPU control is missing or failed')
reference = shap.WORK / 'sequential_reference.npz'
ref = np.load(reference)
checks = []
for name, path, expected in (
    ('background', coalition / 'warmup_bg.npy', ref['p0']),
    ('explain', coalition / 'warmup_explain.npy', ref['p1']),
    ('mixed_8000', coalition / 'predictions.npy', ref['predictions'][0].reshape(-1)),
):
    actual = np.load(path)
    passed = actual.shape == expected.shape and np.allclose(actual, expected, rtol=1e-5, atol=1e-7)
    checks.append({'name': name, 'passed': bool(passed), 'max_abs': float(np.max(np.abs(actual-expected))),
                   'fixture_file': str(path), 'fixture_sha256': sha256(path)})
    if not passed:
        raise ValueError(f'Fixture changes the original prediction: {name}')
stages = {}
for name, folder in (('full', full_short / 'full'), ('short', full_short / 'short'), ('coalition', coalition)):
    path = folder / 'result.json'
    result = json.loads(path.read_text())
    if result['index_sha256'] != sha256(folder / 'index.jsonl'):
        raise ValueError(f'Fixture index changed: {name}')
    if result['predictions_sha256'] != sha256(folder / 'predictions.npy'):
        raise ValueError(f'Fixture predictions changed: {name}')
    if name == 'full' and not result['original_full_baseline']['passed']:
        raise ValueError('Original full baseline failed')
    if name == 'short' and not result['independent_short_reference_matched']:
        raise ValueError('Independent 128-person reference failed')
    stages[name] = {'directory': str(folder), 'result_sha256': sha256(path), 'records': result['records']}
report = {'status': 'validated_diagnostic_controls', 'script_sha256': sha256(Path(__file__)),
          'stages': stages, 'coalition_vs_sequential': checks,
          'control_files': {p.name: sha256(p) for p in (reference, shap.WORK / 'isolated_000.npz', shap.WORK / 'isolated_001.npz')},
          'scope': 'Local diagnostic fixtures and two-person execution controls only; not full SHAP or cross-device validation.'}
save_json(root / 'validated_local_controls.json', report)
print(json.dumps({'status': report['status'], 'checks': checks}, indent=2), flush=True)
