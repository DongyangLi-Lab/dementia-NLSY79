"""Verify the 160 completed mean/complete-case fits from their held-out outputs.

Does not fit models or rewrite comparison predictions. Preserves actual device stamps.
"""
from datetime import datetime
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
from inference.scoring import metrics
from model.specs import MODELS
from train.cohorts import ARTIFACTS, load_cohorts, sha256
from train.imputation import cache_stamp, split_indices

def require(condition, message):
    if not condition:
        raise ValueError(message)

def main():
    dev, _, inputs = load_cohorts(6)
    complete = dev[inputs].notna().all(axis=1).to_numpy()
    stamp = cache_stamp(6, 'final')
    implementation = sha256(ARTIFACTS.parent / 'train/imputation_comparison.py')
    rows = []
    for model in MODELS:
        folder = job_dir('imputation_comparison', 6, 'final', model)
        for strategy in ('mean', 'complete_case'):
            for repeat in range(10):
                key = f'{strategy}-rep{repeat}'
                receipt_path = folder / f'{key}.done.json'
                receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
                saved_stamp = receipt['stamp']
                require(saved_stamp['imputation'] == stamp and saved_stamp['implementation'] == implementation,
                        f'Stale source implementation: {model}/{key}')
                result, raw = folder / f'{key}.csv', folder / f'{key}-predictions.csv'
                require(verified(receipt_path, saved_stamp, (result, raw)), f'Invalid receipt: {model}/{key}')
                table = pd.read_csv(result, float_precision='round_trip')
                require(len(table) == 1, f'Expected one result row: {model}/{key}')
                summary = table.iloc[0]
                require(summary['model'] == model and summary['strategy'] == strategy
                        and summary['repeat'] == repeat and summary['horizon'] == 6, 'Result identity mismatch')
                data = pd.read_csv(raw, float_precision='round_trip')
                test = split_indices(dev.y.to_numpy(), repeat)[2]
                if strategy == 'complete_case':
                    test = test[complete[test]]
                require(np.array_equal(data[['hhidpn', 'y']].to_numpy(),
                                       dev.iloc[test][['hhidpn', 'y']].to_numpy()),
                        f'Wrong held-out cohort/order: {model}/{key}')
                # CSV has no dtype metadata. Restore these estimators' float32
                # probability output before thresholding and Brier computation.
                # GAT explicitly returns float64, as do the classical models.
                probability_dtype = np.float32 if model in ('RealMLP', 'TabICL') else np.float64
                p = data.p.to_numpy(dtype=probability_dtype)
                require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), 'Invalid probability')
                actual = metrics(data.y.to_numpy(), p, float(summary['threshold']))
                errors = {name: abs(float(value) - float(summary[name])) for name, value in actual.items()}
                require(all(np.isclose(value, summary[name], rtol=1e-10, atol=1e-12)
                            for name, value in actual.items()), f'Metrics differ from saved predictions: {model}/{key}')
                rows.append({'model': model, 'strategy': strategy, 'repeat': repeat, 'rows': len(data),
                             'receipt': str(receipt_path), 'receipt_sha256': sha256(receipt_path),
                             'prediction_sha256': sha256(raw), 'result_sha256': sha256(result),
                             'max_metric_difference': max(errors.values()),
                             'probability_dtype': np.dtype(probability_dtype).name,
                             'device': saved_stamp['device']})
        print(f'{model}: 20 comparison outputs, cohort identities and metrics verified', flush=True)
    require(len(rows) == 160, 'Incomplete comparison scope')
    report = {'status': 'validated', 'observed_at': datetime.now().astimezone().isoformat(),
              'scope': '160 saved mean/complete-case held-out outputs, receipts, split identities and recomputed metrics; no refitting or GPU replay',
              'implementation_sha256': sha256(Path(__file__)), 'runs': rows, 'run_count': len(rows)}
    save_json(ARTIFACTS / 'comparison_sources_audit.json', report)
    print('Verified 160 comparison outputs', flush=True)

if __name__ == '__main__':
    main()
