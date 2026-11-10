"""Publish TabICL importance only after both verified intermediate stages finish."""
import importlib.util
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from artifact_store import job_dir, save_json, save_receipt, verified, write_file
from inference import evaluation, importance
from train.cohorts import ARTIFACTS, sha256

HERE = Path(__file__).resolve().parent
def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

def make_tables(perm, groups, phi):
    if len(groups) != 60 or phi.shape != (importance.KS_EXPLAIN, 60) or len(perm) != 60:
        raise ValueError('Incomplete importance results')
    if not np.isfinite(phi).all() or len(set(groups)) != 60 or set(groups) != set(perm.variable):
        raise ValueError('Invalid or misaligned SHAP variables')
    shap = pd.DataFrame({'horizon': 6, 'model': 'TabICL', 'variable': groups,
                         'mean_abs_SHAP': np.abs(phi).mean(axis=0)})
    shap['rank'] = shap['mean_abs_SHAP'].rank(ascending=False, method='min').astype(int)
    shap = shap.sort_values('rank')
    paired = perm.merge(shap, on=['horizon', 'model', 'variable'], validate='one_to_one')
    agree = {'horizon': 6, 'model': 'TabICL', 'method': 'KernelSHAP (approximate)',
             'spearman_rank_correlation': stats.spearmanr(paired.importance_AUPRC_drop, paired.mean_abs_SHAP).correlation,
             'shared_top10': len(set(perm.variable.iloc[:10]) & set(shap.variable.iloc[:10]))}
    if not np.isfinite(agree['spearman_rank_correlation']):
        raise ValueError('Rank agreement is not finite')
    return shap.assign(method=agree['method']), pd.DataFrame([agree])

def main(cross_device=False):
    target = job_dir('importance', 6, 'final', 'TabICL')
    pred = job_dir('predictions', 6, 'final', 'TabICL') / 'inference.done.json'
    stamp = {'inference': sha256(pred), 'implementation': sha256(Path(importance.__file__))}
    names = ('permutation', 'meta', 'shap', 'rank_agreement')
    files = [target / f'{n}.csv' for n in names]
    done = target / 'importance.done.json'
    if verified(done, stamp, files):
        print('TabICL importance receipt already valid', flush=True)
        return
    perm_module = module('permutation_finalize', HERE / 'tabicl_permutation_finalize.py')
    shap_module = module('shap_sharded', HERE / 'tabicl_kernel_shap_sharded.py')
    # These functions validate every source and stop if any stage is incomplete.
    perm_module.aggregate()
    if cross_device:
        from artifacts.aggregate_guarded_tabicl_shap import aggregate
        shap_path = aggregate()
    else:
        shap_module.aggregate()
        shap_path = ARTIFACTS / 'tabicl_kernel_shap_sharded' / 'aggregated.npz'
    perm_path = ARTIFACTS / 'tabicl_bootstrap_precompute' / 'permutation.csv'
    baseline_path = ARTIFACTS / 'tabicl_bootstrap_precompute' / 'baseline.npy'
    perm = pd.read_csv(perm_path)
    data = np.load(shap_path)
    shap, agreement = make_tables(perm, list(data['groups']), data['values'])
    baseline = np.load(baseline_path)
    saved = evaluation.load(6, 'TabICL', 'temporal')
    if not np.allclose(baseline, saved.p.to_numpy(), rtol=1e-5, atol=1e-7):
        raise ValueError('Final baseline failed saved prediction check')
    from sklearn.metrics import average_precision_score
    meta = pd.DataFrame([{'horizon': 6, 'model': 'TabICL',
                         'base_AUPRC': average_precision_score(saved.y.to_numpy(), baseline),
                         'permutation_seconds': np.nan,
                         'max_abs_diff_vs_saved_predictions': np.max(np.abs(baseline-saved.p.to_numpy()))}])
    for frame, path in zip((perm, meta, shap, agreement), files):
        write_file(path, lambda p, f=frame: f.to_csv(p, index=False))
    phi_path = target / 'kernel_shap_values.npz'
    def save_phi(path):
        with path.open('wb') as f:
            np.savez_compressed(f, groups=data['groups'], values=data['values'])
    write_file(phi_path, save_phi)
    provenance = {'status': 'complete', 'implementation': sha256(HERE / 'finalize_tabicl_importance.py'),
                  'permutation_intermediate': sha256(perm_path), 'shap_intermediate': sha256(shap_path),
                  'baseline': sha256(baseline_path), 'kernel_shap_values': sha256(phi_path),
                  'shap_source_kind': 'guarded_cross_device' if cross_device else 'local',
                  'shap_aggregate_receipt': sha256(shap_path.with_suffix('.done.json')),
                  'permutation_seconds': None,
                  'permutation_seconds_note': 'Unavailable: isolated executions, retries and reuse; not estimated.'}
    save_json(target / 'isolated_importance_provenance.json', provenance)
    save_receipt(done, stamp, files)
    print('TabICL importance finalized and receipt saved', flush=True)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cross-device', action='store_true')
    args = parser.parse_args()
    main(args.cross_device)
