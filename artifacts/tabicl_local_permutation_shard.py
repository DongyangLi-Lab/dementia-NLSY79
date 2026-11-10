"""Compute assigned TabICL permutation groups with a fresh process per prediction."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from artifact_store import job_dir, save_json, save_receipt, verified, write_file
from inference import evaluation, importance
from train.cohorts import ARTIFACTS, sha256

SCRIPT = Path(__file__).resolve()
TARGET = job_dir('importance', 6, 'final', 'TabICL')
WORK = ARTIFACTS / 'tabicl_local_permutations'


def inputs():
    model, _, X, _, _, predictors = importance.load_temporal_model('TabICL', 6)
    groups = importance.predictor_groups(predictors)
    return model, X, groups


def save_array(path, values):
    def write(tmp):
        with tmp.open('wb') as stream:
            np.save(stream, values)
    write_file(path, write)


def worker(group_index, replicate):
    model, X, groups = inputs()
    if group_index < 0:
        actual = importance._predict(model, X)
        expected = evaluation.load(6, 'TabICL', 'temporal')['p'].to_numpy()
        maxdiff = float(np.max(np.abs(actual - expected)))
        if not np.allclose(actual, expected, rtol=1e-5, atol=1e-7):
            raise ValueError(f'TabICL temporal baseline mismatch: max_abs={maxdiff}')
        save_json(WORK / 'baseline.json', {'max_abs_diff': maxdiff, 'rows': len(actual),
                                         'worker_sha256': sha256(SCRIPT)})
        print(f'Baseline reproduced: max_abs={maxdiff}', flush=True)
        return
    name, columns = list(groups.items())[group_index]
    rng = np.random.default_rng(np.random.SeedSequence([2026, group_index]))
    for _ in range(replicate + 1):
        order = rng.permutation(len(X))
    Xp = X.copy()
    Xp[columns] = X[columns].to_numpy()[order]
    p = importance._predict(model, Xp)
    save_array(WORK / f'{group_index:02d}-{replicate}.npy', p)
    print(f'{name}: permutation {replicate+1}/5 complete', flush=True)


def main(indices):
    WORK.mkdir(parents=True, exist_ok=True)
    stamp = {'inference': sha256(job_dir('predictions', 6, 'final', 'TabICL') / 'inference.done.json'),
             'implementation': sha256(Path(importance.__file__))}
    provenance = {'indices': indices, 'worker_sha256': sha256(SCRIPT), 'stamp': stamp,
                  'device': os.environ.get('CUDA_VISIBLE_DEVICES'), 'status': 'running'}
    save_json(WORK / 'provenance.json', provenance)
    def run(i, j):
        subprocess.run([sys.executable, '-u', str(SCRIPT), '--worker', str(i), '--replicate', str(j)],
                       cwd=ARTIFACTS.parent, env=os.environ.copy(), check=True)
    run(-1, 0)
    for i in indices:
        raw = TARGET / 'permutations' / f'{i:02d}.npy'
        receipt = raw.with_suffix('.done.json')
        if verified(receipt, stamp, (raw,)):
            print(f'group {i} reused', flush=True)
            continue
        for j in range(5):
            run(i, j)
        arrays = [np.load(WORK / f'{i:02d}-{j}.npy') for j in range(5)]
        save_array(raw, np.asarray(arrays))
        save_receipt(receipt, stamp, (raw,))
        print(f'group {i}: saved', flush=True)
    provenance['status'] = 'complete'
    save_json(WORK / 'provenance.json', provenance)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indices', type=int, nargs='+')
    parser.add_argument('--worker', type=int)
    parser.add_argument('--replicate', type=int, default=0)
    args = parser.parse_args()
    if args.worker is not None:
        worker(args.worker, args.replicate)
    elif args.indices and len(set(args.indices)) == len(args.indices) and all(0 <= i < 60 for i in args.indices):
        main(args.indices)
    else:
        raise ValueError('Specify unique group indices from 0 to 59')
