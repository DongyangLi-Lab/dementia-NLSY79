"""Precompute per-group bootstrap AUPRCs for completed TabICL permutations.

Intermediate only: the frozen-model baseline and final rank table are computed later.
Each receipt ties the result to the exact prediction, cohort, and analysis code.
"""
import concurrent.futures
import hashlib
import os
from pathlib import Path
import sys

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics import average_precision_score
from artifact_store import job_dir, save_receipt, verified, write_file
from inference import evaluation, importance
from train.cohorts import ARTIFACTS, load_cohorts, sha256

TARGET = job_dir('importance', 6, 'final', 'TabICL')
OUT = ARTIFACTS / 'tabicl_bootstrap_precompute'
STAMP = {'inference': sha256(job_dir('predictions', 6, 'final', 'TabICL') / 'inference.done.json'),
         'implementation': sha256(Path(importance.__file__))}
Y = INDICES = COHORT_HASH = None

def init_worker():
    global Y, INDICES, COHORT_HASH
    _, ext, _ = load_cohorts(6)
    Y = ext.y.to_numpy()
    COHORT_HASH = hashlib.sha256(np.asarray(ext.hhidpn, dtype=np.int64).tobytes() + np.asarray(Y, dtype=np.int64).tobytes()).hexdigest()
    INDICES = list(evaluation.cluster_bootstrap_indices(ext.hhidpn.to_numpy(), importance.N_BOOT, seed=2026))

def compute(i):
    raw = TARGET / 'permutations' / f'{i:02d}.npy'
    receipt = raw.with_suffix('.done.json')
    if not verified(receipt, STAMP, (raw,)):
        return f'{i}: prediction missing or receipt invalid'
    stamp = {'permutation': sha256(raw), 'prediction_receipt': sha256(receipt),
             'importance': sha256(Path(importance.__file__)), 'evaluation': sha256(Path(evaluation.__file__)),
             'cohort_ids_labels': COHORT_HASH, 'precompute': sha256(Path(__file__))}
    result = OUT / f'{i:02d}.npz'
    done = OUT / f'{i:02d}.done.json'
    if verified(done, stamp, (result,)):
        return f'{i}: reused'
    predictions = np.load(raw)
    if predictions.shape != (importance.N_PERMUTATIONS, len(Y)) or not np.isfinite(predictions).all():
        raise ValueError(f'{i}: invalid prediction array {predictions.shape}')
    point = np.mean([average_precision_score(Y, p) for p in predictions])
    bootstrap = np.array([np.mean([average_precision_score(Y[idx], p[idx]) for p in predictions]) for idx in INDICES])
    def save(path):
        with path.open('wb') as stream:
            np.savez_compressed(stream, permutation_auprc=point, bootstrap_auprc=bootstrap)
    write_file(result, save)
    save_receipt(done, stamp, (result,))
    return f'{i}: computed {len(bootstrap)} bootstrap samples'

if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = [i for i in range(60) if (TARGET / 'permutations' / f'{i:02d}.done.json').exists()]
    print(f'Checking {len(candidates)} completed prediction groups', flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=4, initializer=init_worker) as pool:
        for message in pool.map(compute, candidates):
            print(message, flush=True)
