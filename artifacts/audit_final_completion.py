"""Require the complete revised computation scope before declaring completion."""
from datetime import datetime
from io import StringIO
import json
import os
from pathlib import Path
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from artifact_store import job_dir, save_json, verified
from artifacts.finish_verified_sources import EXPECTED
from artifacts.parallel_evaluation import input_stamp
from artifacts import tabicl_kernel_shap_sharded as core
from inference import importance, supplementary_tables
from model.specs import MODELS
from train.cohorts import ARTIFACTS, sha256
from train.prepared import require_prepared

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def require(condition, message):
    if not condition:
        raise ValueError(message)

def current_receipt(path):
    data = read(path)
    require(verified(path, data['stamp'], [path.parent / name for name in data['sha256']]),
            f'Receipt/output integrity failed: {path}')

def main():
    source = read(ARTIFACTS / 'source_status.json')
    require(source['state'] == 'complete', 'Final table sources are not complete')
    require(set(source['final_tables']) == set(EXPECTED), 'Wrong final table scope')
    for relative, digest in source['table_files'].items():
        require(sha256(ARTIFACTS / relative) == digest, f'Table changed: {relative}')
    for relative, digest in source['verified_input_receipts'].items():
        require(sha256(ARTIFACTS / relative) == digest, f'Input receipt changed: {relative}')
    require_prepared()
    verification = read(ARTIFACTS / 'verification.json')
    require(all(verification[k] == n for k, n in
                {'imputations': 44, 'model_runs': 32, 'predictors': 60, 'model_inputs': 94}.items()),
            'Unexpected benchmark scope')
    frozen = read(ARTIFACTS / 'frozen_checkpoint_integrity.json')
    require(frozen['status'] == 'validated' and frozen['fit_count'] == 352
            and frozen['implementation_sha256'] == sha256(ARTIFACTS / 'audit_frozen_checkpoint_integrity.py'),
            'Frozen fit audit missing or stale')
    required_runs = {(o, h, m) for o in ('final', 'any') for h in (6, 8) for m in MODELS}
    require({(r['outcome'], r['horizon'], r['model']) for r in frozen['runs']} == required_runs,
            'Missing benchmark runs')
    for run in frozen['runs']:
        folder = job_dir('checkpoints', run['horizon'], run['outcome'], run['model'])
        require(len(run['checkpoint_sha256']) == 11, 'Missing repeat/temporal fit')
        for name, digest in (run['checkpoint_sha256'] | run['receipt_sha256']).items():
            require(sha256(folder / name) == digest, f'Frozen fit changed: {folder / name}')
        for name in run['receipt_sha256']:
            current_receipt(folder / name)
        pred = job_dir('predictions', run['horizon'], run['outcome'], run['model']) / 'inference.done.json'
        require(sha256(pred) == run['inference_receipt_sha256'], 'Prediction receipt changed')
        current_receipt(pred)
    evaluation = read(ARTIFACTS / 'final_evaluation_sources_audit.json')
    require(evaluation['status'] == 'validated' and set(evaluation['outcomes']) == {'final', 'any'},
            'Incomplete evaluation audit')
    for outcome, evidence in evaluation['outcomes'].items():
        require(evidence['stamp'] == input_stamp(outcome), 'Evaluation inputs changed')
        for relative, info in evidence['outputs'].items():
            require(sha256(ARTIFACTS / relative) == info['sha256'], 'Evaluation table changed')
        for cell, digest in evidence['fragment_sha256'].items():
            require(sha256(ARTIFACTS / 'parallel_evaluation' / outcome / f'{cell}.pkl') == digest,
                    'Bootstrap fragment changed')
    comparisons = read(ARTIFACTS / 'comparison_sources_audit.json')
    require(comparisons['status'] == 'validated' and comparisons['run_count'] == 160, 'Comparison incomplete')
    for run in comparisons['runs']:
        receipt = Path(run['receipt'])
        require(sha256(receipt) == run['receipt_sha256'], 'Comparison receipt changed')
        current_receipt(receipt)
    for model in MODELS:
        folder = job_dir('importance', 6, 'final', model)
        current_receipt(folder / 'importance.done.json')
        for name, column in [('permutation', 'importance_AUPRC_drop'), ('shap', 'mean_abs_SHAP')]:
            frame = pd.read_csv(folder / f'{name}.csv')
            require(len(frame) == frame.variable.nunique() == 60 and np.isfinite(frame[column]).all(),
                    f'Incomplete importance: {model}/{name}')
    gate = read(ARTIFACTS / 'tabicl_permutation_execution_audit.json')
    require(gate['status'] == 'validated' and gate['required_groups'] == list(range(16, 29)),
            'Execution reconciliation not validated')
    for field in ('replay_report', 'reconciliation_report'):
        require(sha256(Path(gate[field])) == gate[field + '_sha256'], f'{field} changed')
    for i, digest in gate['validated_raw_sha256'].items():
        require(sha256(job_dir('importance', 6, 'final', 'TabICL') / 'permutations' / f'{int(i):02d}.npy') == digest,
                'Reconciled raw values changed')
    aggregated = core.WORK / 'aggregated_cross_device.npz'
    current_receipt(aggregated.with_suffix('.done.json'))
    provenance = read(core.WORK / 'aggregated_cross_device_sources.json')
    require(provenance['status'] == 'complete' and provenance['core_stamp'] == core.stamp(), 'Stale SHAP aggregate')
    require(set(provenance['subjects']) == set(map(str, range(500))), 'Missing/extra SHAP participants')
    with np.load(aggregated, allow_pickle=False) as data:
        phi, groups = data['values'], data['groups']
    require(phi.shape == (500, 60) and np.isfinite(phi).all() and len(set(groups)) == 60,
            'Invalid final SHAP matrix')
    max_additivity_error = 0.0
    for key, item in provenance['subjects'].items():
        path = Path(item['path'])
        require(sha256(path) == item['sha256'], 'SHAP raw output changed')
        with np.load(path, allow_pickle=False) as raw:
            p0, p1, predictions = raw['p0'], raw['p1'], raw['predictions']
        require(predictions.shape == (1, 800, 10), 'Wrong SHAP computational scale')
        def logit(p):
            p = np.clip(p, 1e-6, 1-1e-6)
            return np.log(p / (1-p))
        i = int(key)
        delta = (logit(p1[i:i+1]) - logit(p0.mean()))[0]
        max_additivity_error = max(max_additivity_error, abs(float(phi[i].sum()) - float(delta)))
    require(max_additivity_error < 1e-10, 'SHAP sum does not reproduce own-execution prediction contrast')
    final_folder = job_dir('importance', 6, 'final', 'TabICL')
    final_proof = read(final_folder / 'isolated_importance_provenance.json')
    require(final_proof['shap_intermediate'] == sha256(aggregated)
            and final_proof['kernel_shap_values'] == sha256(final_folder / 'kernel_shap_values.npz'),
            'Published TabICL source mismatch')
    saved_shap = pd.read_csv(final_folder / 'shap.csv').set_index('variable')
    require(np.allclose(saved_shap.loc[groups, 'mean_abs_SHAP'], np.abs(phi).mean(axis=0),
                        rtol=1e-12, atol=1e-14), 'Published SHAP summary differs from matrix')
    expected_tables = supplementary_tables.build()
    require(set(expected_tables) == set(EXPECTED), 'Table reconstruction scope incomplete')
    for name, frame in expected_tables.items():
        path = ARTIFACTS / 'tables/supplementary' / f'{name}.csv'
        actual = pd.read_csv(path, float_precision='round_trip')
        # Formatting-only numeric strings (for example mean |SHAP|) are inferred
        # as numbers on CSV read; compare the same serialization on both sides.
        reconstructed = pd.read_csv(StringIO(frame.to_csv(index=False)), float_precision='round_trip')
        pd.testing.assert_frame_equal(actual, reconstructed, check_dtype=False,
                                      check_exact=False, rtol=1e-10, atol=1e-12)
    death = pd.read_csv(ARTIFACTS / 'tables/death_exclusion_audit.csv')
    require(len(death) == 8 and death[['accepted_with_wave_death_flag', 'accepted_with_prior_death_year',
            'accepted_with_death_month_before_final_interview']].eq(0).all().all(), 'Death audit violations')
    evidence_names = ['verification.json', 'prepared_inputs.json', 'frozen_checkpoint_integrity.json',
                      'final_evaluation_sources_audit.json', 'comparison_sources_audit.json',
                      'tabicl_permutation_execution_audit.json', 'tabicl_cross_device_acceptance.json',
                      'tabicl_shap_run_inventory.json', 'source_status.json']
    save_json(ARTIFACTS / 'completion_audit.json', {
        'status': 'complete', 'observed_at': datetime.now().astimezone().isoformat(),
        'scope': 'All revised training, held-out verification, evaluation, importance, imputation comparisons and final numerical sources. Figure/document rendering excluded by README.',
        'counts': {'imputation_caches': 44, 'model_runs': 32, 'fitted_checkpoints': 352,
                   'comparison_outputs': 160, 'importance_models': 8, 'tabicl_shap_participants': 500,
                   'shap_groups': 60, 'final_tables': 23},
        'max_shap_additivity_error': max_additivity_error,
        'cross_device_qualification': 'Original strict gate failed; original 500/800/10 computation accepted under recorded per-job engineering checks.',
        'implementation_sha256': sha256(Path(__file__)),
        'evidence_sha256': {name: sha256(ARTIFACTS / name) for name in evidence_names}})
    print('Complete benchmark scope validated: 352 fits, 500 SHAP participants, 23 final tables', flush=True)

if __name__ == '__main__':
    main()
