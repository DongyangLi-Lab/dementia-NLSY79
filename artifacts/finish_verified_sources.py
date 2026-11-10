"""Compile final sources from verified existing results without refitting models."""
from datetime import datetime
import json
import os
from pathlib import Path
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from artifact_store import job_dir, save_json, verified, write_file
from inference import importance, predictor_tables, supplementary_tables
from model.specs import MODELS
from train import imputation_comparison
from train.cohorts import ARTIFACTS, sha256

EXPECTED = ['Table1', 'S1', *(f'S{i}' for i in range(9, 22)),
            'S22a', 'S22b', 'S23', 'S24a', 'S24b', 'S25', 'S26', 'S27']

def require(condition, message):
    if not condition:
        raise ValueError(message)

def main():
    sources = {}
    for model in MODELS:
        folder = job_dir('importance', 6, 'final', model)
        receipt = folder / 'importance.done.json'
        names = ['permutation', 'meta', 'shap', 'rank_agreement']
        if model in ('LR', 'LightGBM'):
            names.append('kernel_shap_check')
        stamp = {'inference': sha256(job_dir('predictions', 6, 'final', model) / 'inference.done.json'),
                 'implementation': sha256(Path(importance.__file__))}
        require(verified(receipt, stamp, [folder / f'{name}.csv' for name in names]),
                f'Incomplete or stale importance: {model}')
        for name in ('permutation', 'shap'):
            frame = pd.read_csv(folder / f'{name}.csv')
            require(len(frame) == 60 and frame.variable.nunique() == 60
                    and frame.model.eq(model).all() and frame.horizon.eq(6).all(),
                    f'Incomplete predictor scope: {model}/{name}')
        sources[str(receipt.relative_to(ARTIFACTS))] = sha256(receipt)
    audit_path = ARTIFACTS / 'comparison_sources_audit.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    require(audit['status'] == 'validated' and audit['run_count'] == 160
            and audit['implementation_sha256'] == sha256(ARTIFACTS / 'audit_comparison_sources.py'),
            'Comparison audit missing or stale')
    expected = {(m, s, r) for m in MODELS for s in ('mean', 'complete_case') for r in range(10)}
    require({(r['model'], r['strategy'], r['repeat']) for r in audit['runs']} == expected,
            'Incomplete comparison audit scope')
    for row in audit['runs']:
        receipt = Path(row['receipt'])
        require(sha256(receipt) == row['receipt_sha256'], 'Comparison receipt changed')
        saved = json.loads(receipt.read_text(encoding='utf-8'))
        require(verified(receipt, saved['stamp'], [receipt.parent / name for name in saved['sha256']]),
                'Comparison output changed after audit')
        sources[str(receipt.relative_to(ARTIFACTS))] = sha256(receipt)
    sources[str(audit_path.relative_to(ARTIFACTS))] = sha256(audit_path)
    # These are CPU table builders. They never call imputation_comparison.run.
    importance.merge()
    imputation_comparison.merge()
    predictor_tables.write()
    tables = supplementary_tables.build()
    require(set(tables) == set(EXPECTED) and len(EXPECTED) == 23, 'Final table set incomplete')
    require(all(len(frame) > 0 for frame in tables.values()), 'An expected final table is empty')
    require(len(tables['S26']) == 8 and tables['S26']['Model'].nunique() == 8,
            'S26 does not cover all eight models')
    out = ARTIFACTS / 'tables/supplementary'
    for name, frame in tables.items():
        write_file(out / f'{name}.csv', lambda path, frame=frame: frame.to_csv(path, index=False))
    files = list((ARTIFACTS / 'tables').glob('*.csv')) + [out / f'{name}.csv' for name in EXPECTED]
    save_json(ARTIFACTS / 'source_status.json', {
        'state': 'complete', 'observed_at': datetime.now().astimezone().isoformat(),
        'scope': 'Revised numerical table and figure data sources; historical partial tables and document/figure rendering excluded',
        'final_tables': {name: {'rows': len(frame), 'sha256': sha256(out / f'{name}.csv')}
                         for name, frame in tables.items()},
        'table_files': {p.relative_to(ARTIFACTS).as_posix(): sha256(p) for p in sorted(files)},
        'verified_input_receipts': sources, 'implementation_sha256': sha256(Path(__file__))})
    print(f'Compiled {len(tables)} final tables from eight models and 160 verified comparison outputs', flush=True)

if __name__ == '__main__':
    main()
