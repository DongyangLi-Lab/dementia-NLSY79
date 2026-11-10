"""Build supplement S21 from the two completed exact-vs-KernelSHAP checks."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from artifact_store import job_dir, save_json, verified, write_file
from inference import importance, supplementary_tables
from train.cohorts import ARTIFACTS, sha256

frames, sources = [], {}
for model in ('LR', 'LightGBM'):
    directory = job_dir('importance', 6, 'final', model)
    pred = job_dir('predictions', 6, 'final', model) / 'inference.done.json'
    files = [directory / f'{name}.csv' for name in ('permutation', 'meta', 'shap', 'rank_agreement', 'kernel_shap_check')]
    stamp = {'inference': sha256(pred), 'implementation': sha256(Path(importance.__file__))}
    receipt = directory / 'importance.done.json'
    if not verified(receipt, stamp, files):
        raise ValueError(f'{model}: importance receipt failed')
    frame = pd.read_csv(files[-1])
    if len(frame) != 1 or frame.at[0, 'model'] != model or frame.at[0, 'horizon'] != 6:
        raise ValueError(f'{model}: invalid KernelSHAP comparison')
    rho, top = frame.at[0, 'spearman_rank_correlation'], frame.at[0, 'shared_top10']
    if not (-1 <= rho <= 1 and 0 <= top <= 10):
        raise ValueError(f'{model}: invalid agreement statistics')
    frames.append(frame)
    sources[model] = {'receipt_sha256': sha256(receipt), 'comparison_sha256': sha256(files[-1])}

checks = pd.concat(frames, ignore_index=True)
supplementary_tables.read = lambda name: checks.copy() if name == 'importance_kernel_shap_check_h6.csv' else (_ for _ in ()).throw(ValueError(name))
table = supplementary_tables.t18c_kernel_check()
output_dir = ARTIFACTS / 'tables' / 'supplementary_partial_verified'
output = output_dir / 'S21.csv'
write_file(output, lambda path: table.to_csv(path, index=False))
manifest_path = output_dir / 'manifest.json'
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
manifest['description'] = 'Formatting from completed verified analyses. Importance-dependent S25/S26 and complete-source manifest remain pending.'
manifest['tables']['S21'] = {'rows': len(table), 'sha256': sha256(output)}
manifest['s21_importance_sources'] = sources
save_json(manifest_path, manifest)
print(table.to_string(index=False))
