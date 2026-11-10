"""Archive contended results and adopt verified exclusive replay for mismatched groups.

CPU only. The historical audit stays unchanged. A dry run is the default.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import psutil
from artifact_store import job_dir, save_json, save_receipt, verified, write_file
from inference import importance
from train.cohorts import ARTIFACTS, sha256

GATE = ARTIFACTS / 'tabicl_permutation_execution_audit.json'
TARGET = job_dir('importance', 6, 'final', 'TabICL') / 'permutations'
BOOT = ARTIFACTS / 'tabicl_bootstrap_precompute'

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def require(condition, message):
    if not condition:
        raise ValueError(message)

def check_idle():
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            names = {Path(a).name for a in (proc.info['cmdline'] or [])}
            require(not names.intersection({'audit_tabicl_permutations.py',
                'tabicl_bootstrap_precompute.py', 'tabicl_pinned_permutations.py'}),
                f'Wait for audit/raw/bootstrap writer PID {proc.pid}')
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

def main(apply=False):
    check_idle()
    gate = read(GATE)
    require(gate['status'] == 'mismatches_require_reconciliation', 'No completed mismatching audit')
    required = list(range(16, 29))
    require(gate['required_groups'] == required, 'Unexpected audit scope')
    report_path = Path(gate['replay_report'])
    report = read(report_path)
    require(sha256(report_path) == gate['replay_report_sha256'], 'Audit report changed')
    require(report['status'] == 'mismatches_require_reconciliation', 'Audit is not terminal')
    folder = report_path.parent
    require(not (folder / 'interference.json').exists(), 'Replay interference evidence exists')
    for key, filename in [('audit_sha256', 'audit_tabicl_permutations.py'),
                          ('worker_sha256', 'tabicl_pinned_permutations.py')]:
        require(report[key] == sha256(ARTIFACTS / filename), f'Source mismatch: {filename}')
    for label in ('before', 'after'):
        base = report['baseline_checks'][label]
        path = folder / f'baseline_{label}.json'
        actual = read(path)
        require(base['sha256'] == sha256(path) and Path(base['path']) == path,
                f'Baseline {label} evidence changed')
        require(all(base[k] == v for k, v in actual.items()), 'Baseline report mismatch')
        require(actual['rows'] == 14123 and 0 <= actual['max_abs_diff'] <= 1e-7
                and actual['worker_sha256'] == report['worker_sha256'], 'Baseline failed')
    require(set(report['groups']) == set(map(str, required)), 'Incomplete replay scope')
    stamp = {'inference': sha256(job_dir('predictions', 6, 'final', 'TabICL') / 'inference.done.json'),
             'implementation': sha256(Path(importance.__file__))}
    replacements, differences = {}, {}
    for i in required:
        path = TARGET / f'{i:02d}.npy'
        require(sha256(path) == report['original_hashes'][str(i)], f'Original {i} changed')
        require(verified(path.with_suffix('.done.json'), stamp, (path,)), f'Receipt {i} invalid')
        old = np.load(path, allow_pickle=False)
        require(old.shape == (5, 14123) and np.isfinite(old).all()
                and ((old >= 0) & (old <= 1)).all(), f'Invalid original {i}')
        checks = report['groups'][str(i)]
        require([x['replicate'] for x in checks] == list(range(5)), 'Incomplete replicates')
        arrays, changes = [], []
        for j, check in enumerate(checks):
            replay = folder / f'{i:02d}-{j}.npy'
            require(sha256(replay) == check['replay_sha256'], f'Replay {i}/{j} changed')
            p = np.load(replay, allow_pickle=False)
            require(p.shape == (14123,) and np.isfinite(p).all()
                    and ((p >= 0) & (p <= 1)).all(), f'Invalid replay {i}/{j}')
            diff = float(np.max(np.abs(p - old[j])))
            passed = bool(np.allclose(p, old[j], rtol=1e-5, atol=1e-7))
            require(diff == check['max_abs'] and passed == check['passed'], 'Comparison changed')
            arrays.append(p)
            if not passed:
                changes.append({'replicate': j, 'max_abs': diff})
        if changes:
            replacements[i] = np.stack(arrays)
            differences[str(i)] = changes
    require(bool(replacements), 'No mismatches to reconcile')
    print(json.dumps({'apply': apply, 'groups_to_replace': differences}), flush=True)
    if not apply:
        return
    check_idle()
    history = ARTIFACTS / 'tabicl_execution_reconciliation' / datetime.now().strftime('%Y%m%d_%H%M%S')
    history.mkdir(parents=True, exist_ok=False)
    backups = []
    for i in replacements:
        for path in (TARGET / f'{i:02d}.npy', TARGET / f'{i:02d}.done.json',
                     BOOT / f'{i:02d}.npz', BOOT / f'{i:02d}.done.json'):
            require(path.exists(), f'Missing history source {path}')
            dest = history / ('raw' if path.parent == TARGET else 'bootstrap') / path.name
            dest.parent.mkdir(exist_ok=True)
            shutil.copy2(path, dest)
            require(sha256(path) == sha256(dest), 'Backup failed')
            backups.append({'original': str(path), 'backup': str(dest), 'sha256': sha256(dest)})
    shutil.copy2(GATE, history / 'original_gate.json')
    shutil.copy2(report_path, history / 'original_report.json')
    evidence = {'status': 'prepared', 'historical_audit_status': report['status'],
                'report': str(report_path), 'report_sha256': sha256(report_path),
                'script_sha256': sha256(Path(__file__)), 'stamp': stamp,
                'differences': differences, 'backups': backups,
                'reason': 'Adopt exclusive replay bracketed by valid frozen-model baselines; preserve original audit and outputs.'}
    save_json(history / 'reconciliation.json', evidence)
    try:
        require(read(GATE) == gate and sha256(report_path) == gate['replay_report_sha256'], 'Audit changed before commit')
        for i in required:
            require(sha256(TARGET / f'{i:02d}.npy') == report['original_hashes'][str(i)], 'Raw changed before commit')
        save_json(GATE, gate | {'status': 'reconciliation_in_progress', 'history': str(history)})
        for i, values in replacements.items():
            path = TARGET / f'{i:02d}.npy'
            def write(tmp):
                with tmp.open('wb') as stream:
                    np.save(stream, values)
            write_file(path, write)
            save_receipt(path.with_suffix('.done.json'), stamp, (path,))
        current = {}
        for i in required:
            path = TARGET / f'{i:02d}.npy'
            require(verified(path.with_suffix('.done.json'), stamp, (path,)), 'Committed receipt invalid')
            values = np.load(path, allow_pickle=False)
            for j in range(5):
                require(np.allclose(values[j], np.load(folder / f'{i:02d}-{j}.npy'),
                                    rtol=1e-5, atol=1e-7), 'Current results not covered by replay')
            current[str(i)] = sha256(path)
        evidence.update(status='validated', validated_raw_sha256=current,
                        bootstrap_refresh_required=sorted(replacements))
        save_json(history / 'reconciliation.json', evidence)
        save_json(GATE, gate | {'status': 'validated', 'validated_raw_sha256': current,
            'validation_basis': 'Exclusive replay adopted for mismatches, original strict audit preserved',
            'reconciliation_report': str(history / 'reconciliation.json'),
            'reconciliation_report_sha256': sha256(history / 'reconciliation.json')})
    except BaseException:
        for item in backups:
            if Path(item['original']).parent == TARGET:
                shutil.copy2(item['backup'], item['original'])
        save_json(GATE, gate)
        evidence['status'] = 'rolled_back'
        save_json(history / 'reconciliation.json', evidence)
        raise
    print(f'Reconciled {sorted(replacements)}; refresh their bootstrap summaries. Evidence: {history}', flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    main(parser.parse_args().apply)
