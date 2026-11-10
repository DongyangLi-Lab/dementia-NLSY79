"""Replay potentially contended predictions on an exclusively reserved GPU.

Never replaces original predictions. A mismatch leaves the finalization gate shut.
"""
import argparse
from datetime import datetime
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import psutil
from artifact_store import job_dir, save_json
from train.cohorts import ARTIFACTS, sha256

SCRIPT = Path(__file__).resolve()
WORKER = SCRIPT.parent / 'tabicl_pinned_permutations.py'
GATE = ARTIFACTS / 'tabicl_permutation_execution_audit.json'
TARGET = job_dir('importance', 6, 'final', 'TabICL') / 'permutations'

def is_gpu_candidate(command):
    text = ' '.join(command).lower()
    return 'tabicl' in text and any(role in text for role in (
        'permutation', 'diagnostic', 'kernel_shap', 'profiler',
        'exclusive_reference', 'memory_path_probe',
    ))


def competitors(exclude_children=False):
    ancestors = {p.pid for p in psutil.Process().parents()} | {os.getpid()}
    if exclude_children:
        ancestors |= {p.pid for p in psutil.Process().children(recursive=True)}
    found = []
    for p in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmd = p.info['cmdline'] or []
            if p.pid not in ancestors and 'python' in (p.info['name'] or '').lower():
                if is_gpu_candidate(cmd):
                    found.append({'pid': p.pid, 'argv': cmd})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return found

def worker(index, repeat, folder):
    spec = importlib.util.spec_from_file_location('original_permutation', WORKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.module.WORK = folder
    module.module.worker(index, repeat)

def main():
    import json
    gate = json.loads(GATE.read_text(encoding='utf-8'))
    live = competitors()
    if live:
        print({'deferred_for_live_processes': live}, flush=True)
        raise SystemExit(2)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise ValueError('Explicit CUDA_VISIBLE_DEVICES=0 required')
    run = ARTIFACTS / 'tabicl_permutation_replay' / datetime.now().strftime('%Y%m%d_%H%M%S')
    run.mkdir(parents=True, exist_ok=False)
    original_hashes = {str(i): sha256(TARGET / f'{i:02d}.npy') for i in gate['required_groups']}
    report = {'status': 'running', 'worker_sha256': sha256(WORKER), 'audit_sha256': sha256(SCRIPT),
              'original_hashes': original_hashes, 'groups': {}, 'directory': str(run),
              'baseline_checks': {}}
    save_json(run / 'report.json', report)
    def execute(i, j=0):
        if competitors():
            raise RuntimeError('Another TabICL process appeared during reserved replay')
        proc = subprocess.Popen([sys.executable, '-u', str(SCRIPT), '--worker', str(i), '--repeat', str(j),
                                 '--directory', str(run)], cwd=ARTIFACTS.parent)
        interference = []
        while proc.poll() is None:
            interference.extend(competitors(exclude_children=True))
            time.sleep(1)
        if proc.returncode:
            raise RuntimeError(f'Replay worker exited {proc.returncode}')
        if interference:
            save_json(run / 'interference.json', interference)
            raise RuntimeError('Concurrent TabICL process detected; replay not accepted')
    def baseline_check(label):
        execute(-1)
        evidence = json.loads((run / 'baseline.json').read_text(encoding='utf-8'))
        if evidence['worker_sha256'] != sha256(WORKER):
            raise RuntimeError('Baseline used an unexpected worker')
        saved = run / f'baseline_{label}.json'
        save_json(saved, evidence)
        report['baseline_checks'][label] = {'path': str(saved), 'sha256': sha256(saved), **evidence}
        save_json(run / 'report.json', report)
    baseline_check('before')
    for i in gate['required_groups']:
        old = np.load(TARGET / f'{i:02d}.npy')
        checks = []
        for j in range(5):
            execute(i, j)
            actual = np.load(run / f'{i:02d}-{j}.npy')
            checks.append({'replicate': j, 'max_abs': float(np.max(np.abs(actual-old[j]))),
                           'passed': bool(np.allclose(actual, old[j], rtol=1e-5, atol=1e-7)),
                           'replay_sha256': sha256(run / f'{i:02d}-{j}.npy')})
        report['groups'][str(i)] = checks
        save_json(run / 'report.json', report)
        print(f'Group {i}: {sum(x["passed"] for x in checks)}/5 matched', flush=True)
    baseline_check('after')
    for i, digest in original_hashes.items():
        if sha256(TARGET / f'{int(i):02d}.npy') != digest:
            raise RuntimeError('Original predictions changed during audit')
    passed = all(x['passed'] for group in report['groups'].values() for x in group)
    report['status'] = 'validated' if passed else 'mismatches_require_reconciliation'
    save_json(run / 'report.json', report)
    gate['status'] = report['status']
    gate['replay_report'] = str(run / 'report.json')
    gate['replay_report_sha256'] = sha256(run / 'report.json')
    gate['validated_raw_sha256'] = original_hashes if passed else {}
    save_json(GATE, gate)
    print(report['status'], flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int)
    parser.add_argument('--repeat', type=int, default=0)
    parser.add_argument('--directory', type=Path)
    args = parser.parse_args()
    if args.worker is not None:
        worker(args.worker, args.repeat, args.directory)
    else:
        main()
