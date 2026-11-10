"""Capture a verified local baseline and assemble precomputed TabICL importance.

Results stay in the intermediate work directory until SHAP and all final-stage
receipts are ready. Use --capture-baseline only when the local GPU is available.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from artifact_store import job_dir, save_receipt, verified, write_file
from inference import evaluation, importance
from train.cohorts import ARTIFACTS, load_cohorts, sha256

SCRIPT = Path(__file__).resolve()
WORK = ARTIFACTS / 'tabicl_bootstrap_precompute'
PRED = job_dir('predictions', 6, 'final', 'TabICL') / 'inference.done.json'
MODEL = job_dir('checkpoints', 6, 'final', 'TabICL') / 'temporal.pkl'
TARGET = job_dir('importance', 6, 'final', 'TabICL')

def baseline_stamp():
    return {'model': sha256(MODEL), 'inference': sha256(PRED),
            'importance': sha256(Path(importance.__file__)), 'capture': sha256(SCRIPT),
            'pinned_execution': sha256(SCRIPT.parent / 'tabicl_pinned_permutations.py')}

def capture_baseline():
    from artifacts.audit_tabicl_permutations import competitors
    if competitors():
        raise RuntimeError('Baseline capture requires an exclusive GPU window')
    from artifacts import tabicl_pinned_permutations  # activate validated full-cohort geometry
    model, _, x, _, _, _ = importance.load_temporal_model('TabICL', 6)
    actual = importance._predict(model, x)
    saved = evaluation.load(6, 'TabICL', 'temporal')['p'].to_numpy()
    if not np.allclose(actual, saved, rtol=1e-5, atol=1e-7):
        raise ValueError(f'Full temporal baseline mismatch: max_abs={np.max(np.abs(actual-saved))}')
    path = WORK / 'baseline.npy'
    def write(tmp):
        with tmp.open('wb') as stream:
            np.save(stream, actual)
    write_file(path, write)
    save_receipt(path.with_suffix('.done.json'), baseline_stamp(), (path,))
    print(f'Full baseline verified: {len(actual)} rows; max_abs={np.max(np.abs(actual-saved))}', flush=True)

def summarize(y, ids, baseline, groups, statistics):
    """Same point estimate, bootstrap order, percentiles and ranking as importance.py."""
    baseline_auprc = average_precision_score(y, baseline)
    base_boot = np.array([average_precision_score(y[idx], baseline[idx])
                          for idx in evaluation.cluster_bootstrap_indices(ids, importance.N_BOOT, seed=2026)])
    rows = []
    for (name, cols), (point, bootstrap) in zip(groups.items(), statistics, strict=True):
        if len(bootstrap) != importance.N_BOOT:
            raise ValueError('Wrong number of bootstrap replicates')
        lo, hi = np.percentile(base_boot - bootstrap, [2.5, 97.5])
        rows.append({'horizon': 6, 'model': 'TabICL', 'variable': name,
                     'importance_AUPRC_drop': baseline_auprc - float(point),
                     'ci_low': lo, 'ci_high': hi, 'n_inputs': len(cols)})
    result = pd.DataFrame(rows)
    result['rank'] = result['importance_AUPRC_drop'].rank(ascending=False, method='min').astype(int)
    return result.sort_values('rank'), baseline_auprc

def aggregate():
    audit_path = ARTIFACTS / 'tabicl_permutation_execution_audit.json'
    if not audit_path.exists():
        raise ValueError('Permutation execution audit is missing')
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    if audit.get('status') != 'validated':
        raise ValueError('GPU contention replay audit has not passed')
    for i in audit.get('required_groups', []):
        raw = TARGET / 'permutations' / f'{i:02d}.npy'
        if audit.get('validated_raw_sha256', {}).get(str(i)) != sha256(raw):
            raise ValueError(f'Execution audit does not cover current permutation {i}')
    baseline_path = WORK / 'baseline.npy'
    if not verified(baseline_path.with_suffix('.done.json'), baseline_stamp(), (baseline_path,)):
        raise ValueError('Verified local baseline is missing or stale')
    _, ext, predictors = load_cohorts(6)
    saved = evaluation.load(6, 'TabICL', 'temporal')
    y, ids = ext.y.to_numpy(), ext.hhidpn.to_numpy()
    baseline = np.load(baseline_path)
    if not (np.array_equal(ids, saved.hhidpn.to_numpy()) and np.array_equal(y, saved.y.to_numpy())
            and np.allclose(baseline, saved.p.to_numpy(), rtol=1e-5, atol=1e-7)):
        raise ValueError('Baseline/cohort identity verification failed')
    cohort = hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes() + np.asarray(y, dtype=np.int64).tobytes()).hexdigest()
    raw_stamp = {'inference': sha256(PRED), 'implementation': sha256(Path(importance.__file__))}
    groups = importance.predictor_groups(predictors)
    stats = []
    sources = hashlib.sha256()
    for i in range(len(groups)):
        raw = TARGET / 'permutations' / f'{i:02d}.npy'
        raw_receipt = raw.with_suffix('.done.json')
        if not verified(raw_receipt, raw_stamp, (raw,)):
            raise ValueError(f'Missing or stale permutation {i}')
        source = WORK / f'{i:02d}.npz'
        source_stamp = {'permutation': sha256(raw), 'prediction_receipt': sha256(raw_receipt),
                        'importance': sha256(Path(importance.__file__)), 'evaluation': sha256(Path(evaluation.__file__)),
                        'cohort_ids_labels': cohort, 'precompute': sha256(SCRIPT.parent / 'tabicl_bootstrap_precompute.py')}
        if not verified(source.with_suffix('.done.json'), source_stamp, (source,)):
            raise ValueError(f'Missing or stale bootstrap group {i}')
        item = np.load(source)
        stats.append((item['permutation_auprc'], item['bootstrap_auprc']))
        sources.update(bytes.fromhex(sha256(source)))
    result, base_auprc = summarize(y, ids, baseline, groups, stats)
    output = WORK / 'permutation.csv'
    write_file(output, lambda p: result.to_csv(p, index=False))
    save_receipt(output.with_suffix('.done.json'), baseline_stamp() | {'baseline': sha256(baseline_path),
                 'bootstrap_sources_sha256': sources.hexdigest()}, (output,))
    print(f'Prepared {len(result)} permutation rows; baseline AUPRC={base_auprc:.8f}', flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture-baseline', action='store_true')
    parser.add_argument('--aggregate', action='store_true')
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    if args.capture_baseline:
        capture_baseline()
    elif args.aggregate:
        aggregate()
    else:
        parser.error('Choose --capture-baseline or --aggregate')
