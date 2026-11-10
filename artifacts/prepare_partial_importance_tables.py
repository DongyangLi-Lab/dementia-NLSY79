"""Validate and format completed models' S25/S26 rows as intermediate only."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from artifact_store import job_dir, save_json, verified, write_file
from inference import importance, supplementary_tables
from model.specs import MODELS
from train.cohorts import ARTIFACTS, sha256

models = [m for m in MODELS if m != 'TabICL']
frames = {n: [] for n in ('permutation', 'shap', 'rank_agreement')}
sources = {}
for model in models:
    directory = job_dir('importance', 6, 'final', model)
    pred = job_dir('predictions', 6, 'final', model) / 'inference.done.json'
    names = ['permutation', 'meta', 'shap', 'rank_agreement']
    if model in ('LR', 'LightGBM'):
        names.append('kernel_shap_check')
    files = [directory / f'{name}.csv' for name in names]
    stamp = {'inference': sha256(pred), 'implementation': sha256(Path(importance.__file__))}
    receipt = directory / 'importance.done.json'
    if not verified(receipt, stamp, files):
        raise ValueError(f'{model}: importance receipt failed')
    for name in frames:
        frame = pd.read_csv(directory / f'{name}.csv')
        if set(frame.model) != {model} or set(frame.horizon) != {6}:
            raise ValueError(f'{model}: invalid {name} rows')
        expected = 1 if name == 'rank_agreement' else 60
        if len(frame) != expected or (expected == 60 and frame.variable.nunique() != 60):
            raise ValueError(f'{model}: invalid {name} row count')
        frames[name].append(frame)
    sources[model] = sha256(receipt)

def read(name):
    mapping = {'importance_permutation_h6.csv': 'permutation', 'importance_shap_h6.csv': 'shap',
               'importance_rank_agreement_h6.csv': 'rank_agreement'}
    return pd.concat(frames[mapping[name]], ignore_index=True)

supplementary_tables.read = read
table26, table25 = supplementary_tables.t18_shap()
out = ARTIFACTS / 'tables' / 'importance_seven_models_intermediate'
out.mkdir(parents=True, exist_ok=True)
tables = {'S25_seven_models': table25, 'S26_seven_models': table26}
manifest = {'status': 'partial; TabICL pending', 'importance_receipts': sources,
            'formatting_sha256': sha256(Path(supplementary_tables.__file__)), 'tables': {}}
for name, table in tables.items():
    path = out / f'{name}.csv'
    write_file(path, lambda p, t=table: t.to_csv(p, index=False))
    manifest['tables'][name] = {'rows': len(table), 'sha256': sha256(path)}
save_json(out / 'manifest.json', manifest)
print('Prepared intermediate rows:', {name: len(table) for name, table in tables.items()})
