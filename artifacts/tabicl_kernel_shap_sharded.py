"""Resumable original-design TabICL KernelSHAP, gated by sequential equivalence.

No results are published by this tool. Before workers are allowed, GPU validation
must compare two sequential subjects with freshly loaded/warmed isolated workers.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from artifact_store import job_dir, save_receipt, verified, write_file
from inference import importance
from train.cohorts import ARTIFACTS, sha256
from train.imputation import cache_path

SCRIPT = Path(__file__).resolve()
DESIGN = ARTIFACTS / 'tabicl_kernel_shap_design.npz'
WORK = ARTIFACTS / 'tabicl_kernel_shap_sharded'
MODEL = job_dir('checkpoints', 6, 'final', 'TabICL') / 'temporal.pkl'
PRED = job_dir('predictions', 6, 'final', 'TabICL') / 'inference.done.json'

def stamp():
    return {'model': sha256(MODEL), 'inference': sha256(PRED), 'importance': sha256(Path(importance.__file__)),
            'design': sha256(DESIGN), 'worker': sha256(SCRIPT),
            'exclusive_guard': sha256(SCRIPT.parent / 'audit_tabicl_permutations.py')}

def load_inputs():
    design_stamp = {'cache': sha256(cache_path(6, 'final', 'external')),
                    'importance': sha256(Path(importance.__file__)),
                    'design': sha256(SCRIPT.parent / 'tabicl_kernel_shap_design.py'),
                    'inference': sha256(PRED)}
    if not verified(DESIGN.with_suffix('.done.json'), design_stamp, (DESIGN,)):
        raise ValueError('KernelSHAP design receipt is missing or stale')
    model, dev, ext, _, _, groups = importance.load_temporal_model('TabICL', 6)
    design = np.load(DESIGN)
    # NPZ Unicode arrays yield numpy.str_; sklearn recognizes feature names only
    # when every label is a Python str, as in the original cohort DataFrame.
    cols = [str(column) for column in design['columns']]
    if cols != list(ext.columns) or cols != list(dev.columns):
        raise ValueError('Design and imputed data columns differ')
    current_groups = importance.predictor_groups(groups)
    if list(design['groups']) != list(current_groups):
        raise ValueError('Design and predictor groups differ')
    if design['column_mask'].shape != (importance.KS_COALITIONS, len(cols)):
        raise ValueError('Unexpected coalition shape')
    return model, dev.to_numpy(np.float32), ext.to_numpy(np.float32), cols, design

def run_subjects(model, dev, ext, cols, design, indices, predict=importance._predict):
    """Preserve original warm-up, then execute each 8,000-row subject in order."""
    bg = dev[design['background_indices']]
    xe = ext[design['explain_indices']]
    colmask = design['column_mask']
    p0 = predict(model, pd.DataFrame(bg, columns=cols))
    p1 = predict(model, pd.DataFrame(xe, columns=cols))
    outputs = []
    for i in indices:
        a = np.where(colmask[:, None, :], xe[i][None, None, :], bg[None, :, :])
        p = predict(model, pd.DataFrame(a.reshape(-1, len(cols)), columns=cols))
        outputs.append(np.asarray(p).reshape(importance.KS_COALITIONS, importance.KS_BACKGROUND))
    return np.asarray(p0), np.asarray(p1), np.asarray(outputs)

def save_array(path, p0, p1, values):
    def write(tmp):
        with tmp.open('wb') as f:
            np.savez_compressed(f, p0=p0, p1=p1, predictions=values)
    write_file(path, write)

def run_worker(indices, name):
    model, dev, ext, cols, design = load_inputs()
    p0, p1, values = run_subjects(model, dev, ext, cols, design, indices)
    path = WORK / f'{name}.npz'
    save_array(path, p0, p1, values)
    save_receipt(path.with_suffix('.done.json'), stamp() | {'indices': indices}, (path,))
    print(f'{name}: saved subjects {indices}', flush=True)

def check_control():
    reference = WORK / 'sequential_reference.npz'
    if not verified(reference.with_suffix('.done.json'), stamp() | {'indices': [0, 1]}, (reference,)):
        return False
    ref = np.load(reference)
    for i in (0, 1):
        path = WORK / f'isolated_{i:03d}.npz'
        if not verified(path.with_suffix('.done.json'), stamp() | {'indices': [i]}, (path,)):
            return False
        isolated = np.load(path)
        for key, expected in (('p0', ref['p0']), ('p1', ref['p1']), ('predictions', ref['predictions'][i:i+1])):
            if not np.allclose(isolated[key], expected, rtol=1e-5, atol=1e-7):
                print(f'Control mismatch subject={i} field={key} max_abs={np.max(np.abs(isolated[key]-expected))}', flush=True)
                return False
    return True

def spawn(*arguments):
    subprocess.run([sys.executable, '-u', str(SCRIPT), *map(str, arguments)], cwd=ARTIFACTS.parent, check=True)

def aggregate():
    if not check_control():
        raise ValueError('Sequential equivalence control is missing or failed')
    ref = np.load(WORK / 'sequential_reference.npz')
    design = np.load(DESIGN)
    p0, p1 = ref['p0'], ref['p1']
    results = []
    base_stamp = stamp()
    source_hash = hashlib.sha256()
    for i in range(importance.KS_EXPLAIN):
        path = WORK / f'subject_{i:03d}.npz'
        if not verified(path.with_suffix('.done.json'), base_stamp | {'indices': [i]}, (path,)):
            raise ValueError(f'Missing or stale subject {i}')
        source_hash.update(bytes.fromhex(sha256(path)))
        item = np.load(path)
        if not (np.allclose(item['p0'], p0, rtol=1e-5, atol=1e-7)
                and np.allclose(item['p1'], p1, rtol=1e-5, atol=1e-7)):
            raise ValueError(f'Warm-up mismatch for subject {i}')
        predictions = item['predictions']
        if predictions.shape != (1, importance.KS_COALITIONS, importance.KS_BACKGROUND):
            raise ValueError(f'Invalid prediction shape for subject {i}')
        results.append(predictions[0])
    logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    v0 = logit(p0.mean())
    v1 = logit(p1)
    values = logit(np.asarray(results).mean(axis=2)).T
    z = design['coalitions'].astype(float)
    d = z[:, :-1] - z[:, [-1]]
    y = (values - v0) - z[:, [-1]] * (v1 - v0)[None, :]
    phi = np.linalg.lstsq(d, y, rcond=None)[0].T
    phi = np.column_stack([phi, (v1 - v0) - phi.sum(axis=1)])
    output = WORK / 'aggregated.npz'
    def write(path):
        with path.open('wb') as stream:
            np.savez_compressed(stream, groups=design['groups'], values=phi)
    write_file(output, write)
    save_receipt(output.with_suffix('.done.json'), base_stamp | {'subjects': importance.KS_EXPLAIN,
                 'subjects_sha256': source_hash.hexdigest(),
                 'sequential_control': sha256(WORK / 'sequential_reference.npz'),
                 'isolated_controls': [sha256(WORK / f'isolated_{i:03d}.npz') for i in (0, 1)]}, (output,))
    print(f'Aggregated SHAP values {phi.shape}; final importance table remains pending', flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--control', action='store_true')
    parser.add_argument('--worker', type=int)
    parser.add_argument('--sequential-reference', action='store_true')
    parser.add_argument('--isolated-control', type=int, choices=(0, 1))
    parser.add_argument('--indices', nargs='+', type=int)
    parser.add_argument('--aggregate', action='store_true')
    args = parser.parse_args()
    if not args.aggregate:
        from artifacts.audit_tabicl_permutations import competitors
        live = competitors()
        if live:
            raise RuntimeError(f'GPU execution deferred: another TabICL process is live: {live}')
    WORK.mkdir(parents=True, exist_ok=True)
    if args.aggregate:
        aggregate()
    elif args.sequential_reference:
        run_worker([0, 1], 'sequential_reference')
    elif args.isolated_control is not None:
        run_worker([args.isolated_control], f'isolated_{args.isolated_control:03d}')
    elif args.control:
        spawn('--sequential-reference')
        for i in (0, 1):
            spawn('--isolated-control', i)
        if not check_control():
            raise ValueError('Sequential versus isolated KernelSHAP control failed')
        print('Sequential versus isolated KernelSHAP control passed', flush=True)
    elif args.worker is not None:
        if not check_control():
            raise ValueError('Sequential equivalence control is missing or failed')
        if not 0 <= args.worker < importance.KS_EXPLAIN:
            raise ValueError('Invalid participant index')
        run_worker([args.worker], f'subject_{args.worker:03d}')
    elif args.indices and len(set(args.indices)) == len(args.indices) and all(0 <= i < importance.KS_EXPLAIN for i in args.indices):
        if not check_control():
            raise ValueError('Sequential equivalence control is missing or failed')
        base_stamp = stamp()
        for i in args.indices:
            path = WORK / f'subject_{i:03d}.npz'
            if not verified(path.with_suffix('.done.json'), base_stamp | {'indices': [i]}, (path,)):
                spawn('--worker', i)
            else:
                print(f'subject {i}: reused', flush=True)
    else:
        parser.error('Select --control or unique --indices')
