"""Complete the reserved local slice with bounded fresh-process stream batches."""
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from artifact_store import save_json, verified
from artifacts import tabicl_kernel_shap_sharded as core
from artifacts.audit_tabicl_permutations import competitors
from train.cohorts import ARTIFACTS, sha256

def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or competitors():
        raise RuntimeError('Requires exclusive GPU0')
    allocation_path = ARTIFACTS / 'shap_dispatch_ownership.json'
    allocation = json.loads(allocation_path.read_text())
    if allocation['local']['inclusive_indices'] != [0, 24] or allocation['unc']['inclusive_indices'] != [25, 499]:
        raise ValueError('Unexpected ownership')
    stamp = core.stamp()
    if not core.check_control():
        raise ValueError('Missing valid local controls')
    driver = ARTIFACTS / 'tabicl_kernel_shap_stream.py'
    driver_sha = sha256(driver)
    def valid(i):
        path = core.WORK / f'subject_{i:03d}.npz'
        return verified(path.with_suffix('.done.json'), stamp | {'indices': [i]}, (path,))
    todo = [i for i in range(25) if not valid(i)]
    status_path = ARTIFACTS / 'tabicl_local_shap_recovery.json'
    state = {'state': 'running', 'supervisor_pid': os.getpid(),
             'reason': 'Bounded batches of four requested subjects, fresh process each; original controls replayed on every startup to limit CPU allocation accumulation.',
             'remaining_at_start': todo, 'core_stamp': stamp, 'stream_driver_sha256': driver_sha,
             'implementation_sha256': sha256(Path(__file__)), 'batches': []}
    def update():
        state['observed_at'] = datetime.now().astimezone().isoformat()
        save_json(status_path, state)
    allocation['local'].update(state='running', supervisor_pid=os.getpid(), recovery_file=str(status_path))
    save_json(allocation_path, allocation)
    update()
    try:
        for offset in range(0, len(todo), 4):
            indices = todo[offset:offset+4]
            if core.stamp() != stamp or sha256(driver) != driver_sha:
                raise ValueError('Execution sources changed')
            log = ARTIFACTS / 'logs' / f'local-shap-fresh-{indices[0]:03d}-{indices[-1]:03d}.log'
            with log.open('a', encoding='utf-8') as output:
                proc = subprocess.Popen([sys.executable, '-u', str(driver), '--indices', *map(str, indices)],
                                        cwd=ARTIFACTS.parent, env=os.environ.copy(), stdout=output, stderr=subprocess.STDOUT)
                batch = {'indices': indices, 'pid': proc.pid, 'log': str(log)}
                state['batches'].append(batch)
                update()
                batch['exit_code'] = proc.wait()
            update()
            if batch['exit_code'] != 0 or not all(valid(i) for i in indices):
                raise RuntimeError(f'Batch failed: {batch}')
            print(f'Fresh batch {indices} complete', flush=True)
        if not all(valid(i) for i in range(25)) or core.stamp() != stamp:
            raise ValueError('Local slice incomplete or sources changed')
        state['state'] = 'complete'
        allocation = json.loads(allocation_path.read_text())
        allocation['local']['state'] = 'complete'
        save_json(allocation_path, allocation)
    except BaseException as exc:
        state.update(state='needs_attention', error=repr(exc))
        raise
    finally:
        update()

if __name__ == '__main__':
    main()
