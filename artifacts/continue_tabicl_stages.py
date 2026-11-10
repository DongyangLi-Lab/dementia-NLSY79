"""Continue dependent local checks after the current permutation scheduler exits.

Never retries a failed stage. Formal local SHAP requires the opt-in flag and
the explicit, non-overlapping local/UNC allocation file.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import psutil
from artifact_store import job_dir, save_json, verified
from inference import importance
from train.cohorts import ARTIFACTS, sha256

STATE = ARTIFACTS / 'tabicl_next_stages_status.json'

def main(pid, start_local_shap=False):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise ValueError('Explicit GPU0 assignment required')
    owner = psutil.Process(pid)
    argv = owner.cmdline()
    if not any(Path(a).name == 'tabicl_pinned_permutations.py' for a in argv) or '--indices' not in argv:
        raise ValueError('Wait target is not the expected permutation scheduler')
    state = {'state': 'waiting_for_permutations', 'wait_pid': pid,
             'wait_created': owner.create_time(), 'wait_argv': argv,
             'stages': [], 'script_sha256': sha256(Path(__file__))}
    def update(status):
        state['state'] = status
        state['observed_at'] = datetime.now().astimezone().isoformat()
        save_json(STATE, state)
        print(status, flush=True)
    update('waiting_for_permutations')
    try:
        while owner.is_running():
            time.sleep(5)
        target = job_dir('importance', 6, 'final', 'TabICL') / 'permutations'
        stamp = {'inference': sha256(job_dir('predictions', 6, 'final', 'TabICL') / 'inference.done.json'),
                 'implementation': sha256(Path(importance.__file__))}
        missing = [i for i in range(60) if not verified(target / f'{i:02d}.done.json', stamp,
                                                       (target / f'{i:02d}.npy',))]
        if missing:
            raise ValueError(f'Scheduler stopped with missing/invalid groups: {missing}')
        steps = [
            ('execution_replay', ['audit_tabicl_permutations.py']),
            ('bootstrap_refresh', ['tabicl_bootstrap_precompute.py']),
            ('baseline_capture', ['tabicl_permutation_finalize.py', '--capture-baseline']),
            ('shap_control', ['tabicl_kernel_shap_sharded.py', '--control']),
        ]
        for name, args in steps:
            if name == 'shap_control':
                from artifacts.tabicl_kernel_shap_sharded import check_control
                if check_control():
                    state['stages'].append({'name': name, 'reused_verified_control': True})
                    continue
            update(name)
            log = ARTIFACTS / 'logs' / f'tabicl-next-{name}.log'
            with log.open('a', encoding='utf-8') as stream:
                subprocess.run([sys.executable, '-u', str(ARTIFACTS / args[0]), *args[1:]],
                               cwd=ARTIFACTS.parent, stdout=stream, stderr=subprocess.STDOUT, check=True)
            state['stages'].append({'name': name, 'log': str(log), 'exit_code': 0})
            if name == 'execution_replay':
                state['execution_audit'] = json.loads((ARTIFACTS / 'tabicl_permutation_execution_audit.json').read_text())['status']
        if start_local_shap:
            allocation_path = ARTIFACTS / 'shap_dispatch_ownership.json'
            allocation = json.loads(allocation_path.read_text(encoding='utf-8'))
            if (allocation['local']['inclusive_indices'] != [0, 24]
                    or allocation['unc']['inclusive_indices'] != [25, 499]
                    or allocation['design_sha256'] != sha256(ARTIFACTS / 'tabicl_kernel_shap_design.npz')):
                raise ValueError('SHAP allocation or design changed; do not start the reserved slice')
            update('local_shap_stream')
            log = ARTIFACTS / 'logs' / 'tabicl-next-local_shap_stream.log'
            allocation['local']['state'] = 'running'
            allocation['local']['supervisor_pid'] = os.getpid()
            save_json(allocation_path, allocation)
            with log.open('a', encoding='utf-8') as stream:
                subprocess.run([sys.executable, '-u', str(ARTIFACTS / 'tabicl_kernel_shap_stream.py'),
                                '--indices', *map(str, range(25))], cwd=ARTIFACTS.parent,
                               stdout=stream, stderr=subprocess.STDOUT, check=True)
            allocation['local']['state'] = 'complete'
            save_json(allocation_path, allocation)
            state['stages'].append({'name': 'local_shap_stream', 'indices': [0, 24], 'log': str(log), 'exit_code': 0})
            update('local_shap_slice_complete')
        else:
            update('shap_control_passed_pending_allocation_review')
    except Exception as exc:
        state['error'] = repr(exc)
        update('needs_attention')
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-pid', type=int, required=True)
    parser.add_argument('--start-local-shap', action='store_true')
    args = parser.parse_args()
    main(args.wait_pid, args.start_local_shap)
