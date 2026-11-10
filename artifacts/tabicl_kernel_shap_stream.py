"""Checkpoint the original sequential KernelSHAP predictions one person at a time.

Uses the existing run_subjects implementation unchanged. The model and the two
warm-up predictions are reused within one reserved GPU run. Subjects 0 and 1 are
always replayed first against the verified sequential control, even on resume.
"""
import argparse
from datetime import datetime
import hashlib
import os
from pathlib import Path
import sys
import threading
import time
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from artifact_store import save_json, save_receipt, verified
from artifacts.audit_tabicl_permutations import competitors
from artifacts import tabicl_kernel_shap_sharded as core
from train.cohorts import sha256

SCRIPT = Path(__file__).resolve()

def main(indices):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise ValueError('This local runner is reserved for GPU0')
    live = competitors()
    if live:
        raise RuntimeError(f'GPU execution deferred: {live}')
    if not core.check_control():
        raise ValueError('Sequential/isolated GPU control must pass before streaming')
    base_stamp = core.stamp()
    requested = set(indices)
    todo = []
    for i in indices:
        path = core.WORK / f'subject_{i:03d}.npz'
        if not verified(path.with_suffix('.done.json'), base_stamp | {'indices': [i]}, (path,)):
            todo.append(i)
    if not todo:
        print('All assigned subjects already have valid receipts', flush=True)
        return
    sequence = [0, 1] + [i for i in todo if i not in (0, 1)]
    folder = core.WORK / 'stream_runs'
    folder.mkdir(exist_ok=True)
    run = folder / f'{datetime.now():%Y%m%d_%H%M%S_%f}.json'
    state = {'status': 'running', 'requested_indices': indices, 'sequence': sequence,
             'core_stamp': base_stamp, 'stream_driver_sha256': sha256(SCRIPT),
             'control_reference_sha256': sha256(core.WORK / 'sequential_reference.npz'),
             'probes': {}, 'subjects': {}, 'started_at': datetime.now().astimezone().isoformat()}
    save_json(run, state)
    stop = threading.Event()
    interference = []
    def watch():
        while not stop.wait(1):
            interference.extend(competitors())
    monitor = threading.Thread(target=watch, daemon=True)
    monitor.start()
    reference = np.load(core.WORK / 'sequential_reference.npz')
    warm = []
    call = 0
    start = time.monotonic()
    def predict(model, frame):
        nonlocal call
        if interference or competitors():
            raise RuntimeError('Concurrent TabICL GPU execution detected')
        began = time.monotonic()
        result = core.importance._predict(model, frame)
        seconds = time.monotonic() - began
        if interference or competitors():
            raise RuntimeError('Concurrent TabICL GPU execution detected during prediction')
        if call < 2:
            key = ('p0', 'p1')[call]
            if result.shape != reference[key].shape or not np.allclose(result, reference[key], rtol=1e-5, atol=1e-7):
                raise ValueError(f'Warm-up differs from verified reference: {key}')
            warm.append(result)
            print(f'{key}: reference matched, {seconds:.2f}s', flush=True)
        else:
            i = sequence[call - 2]
            if i in (0, 1):
                expected = reference['predictions'][i].reshape(-1)
                if result.shape != expected.shape or not np.allclose(result, expected, rtol=1e-5, atol=1e-7):
                    raise ValueError(f'Streaming execution differs from sequential control: subject {i}')
                state['probes'][str(i)] = {'passed': True, 'max_abs': float(np.max(np.abs(result-expected)))}
            if i in requested:
                path = core.WORK / f'subject_{i:03d}.npz'
                receipt = path.with_suffix('.done.json')
                reused = verified(receipt, base_stamp | {'indices': [i]}, (path,))
                if not reused:
                    values = result.reshape(1, core.importance.KS_COALITIONS, core.importance.KS_BACKGROUND)
                    core.save_array(path, warm[0], warm[1], values)
                    save_receipt(receipt, base_stamp | {'indices': [i]}, (path,))
                state['subjects'][str(i)] = {'prediction_sha256': sha256(path), 'receipt_sha256': sha256(receipt),
                                             'seconds': seconds, 'reused': reused}
                save_json(run, state)
                print(f'subject {i}: {"reused" if reused else "saved"}, {seconds:.2f}s', flush=True)
        call += 1
        return result
    try:
        model, dev, ext, cols, design = core.load_inputs()
        core.run_subjects(model, dev, ext, cols, design, sequence, predict=predict)
        if core.stamp() != base_stamp or sha256(SCRIPT) != state['stream_driver_sha256']:
            raise RuntimeError('Execution sources changed during streaming')
        if interference:
            raise RuntimeError('Concurrent TabICL GPU execution was detected')
        state['status'] = 'complete'
        state['elapsed_seconds'] = time.monotonic() - start
        save_json(run, state)
    except BaseException as exc:
        state['status'] = 'failed'
        state['error'] = repr(exc)
        state['interference'] = interference
        save_json(run, state)
        raise
    finally:
        stop.set()
        monitor.join(timeout=3)
        reference.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indices', nargs='+', type=int, required=True)
    args = parser.parse_args()
    if len(args.indices) != len(set(args.indices)) or not all(0 <= i < core.importance.KS_EXPLAIN for i in args.indices):
        parser.error('Provide unique participant indices from 0 through 499')
    main(args.indices)
