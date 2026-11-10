"""Refresh the shared monitor from verified receipts and live local processes."""
from datetime import datetime
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psutil
from artifact_store import job_dir, save_json, verified
from inference import importance
from train.cohorts import ARTIFACTS, sha256
from artifacts.audit_tabicl_permutations import is_gpu_candidate
from artifacts import tabicl_kernel_shap_sharded as shap_stage

target = job_dir('importance', 6, 'final', 'TabICL')
pred = job_dir('predictions', 6, 'final', 'TabICL') / 'inference.done.json'
stamp = {'inference': sha256(pred), 'implementation': sha256(Path(importance.__file__))}
complete, pending = [], []
for i in range(60):
    path = target / 'permutations' / f'{i:02d}.npy'
    (complete if verified(path.with_suffix('.done.json'), stamp, (path,)) else pending).append(i)
processes = []
stage_processes = []
for proc in psutil.process_iter(['pid', 'ppid', 'name', 'cmdline']):
    try:
        cmd = proc.info.get('cmdline') or []
        name = proc.info.get('name') or ''
        if 'python' in name.lower() and any(any(n in c.lower() for n in ('tabicl_local_permutation_shard.py', 'tabicl_pinned_permutations.py')) for c in cmd):
            processes.append({'pid': proc.info['pid'], 'parent_pid': proc.info['ppid'], 'argv': cmd})
        elif 'python' in name.lower() and (is_gpu_candidate(cmd) or any(Path(c).name in (
            'continue_tabicl_stages.py', 'audit_tabicl_permutations.py',
            'tabicl_permutation_finalize.py', 'tabicl_kernel_shap_sharded.py',
            'tabicl_invalid_probability_diagnostic.py') for c in cmd)):
            stage_processes.append({'pid': proc.info['pid'], 'parent_pid': proc.info['ppid'], 'argv': cmd})
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
status = {'state': 'running' if processes or stage_processes else 'needs_attention', 'phase': 'postprocessing',
          'other_local_stage_processes': stage_processes,
          'observed_at': datetime.now().astimezone().isoformat(timespec='seconds'),
          'scope': 'Local process and hash receipts observed at this timestamp; UNC status requires its own live check.',
          'verification': 'artifacts/verification.json',
          'training_and_predictions': '32 model/outcome/horizon runs verified; see verification.json',
          'tabicl_permutation': {'groups_verified': complete, 'groups_pending': pending,
                                 'verified_count': len(complete), 'total': 60, 'local_processes': processes,
                                 'verification_scope': 'File/hash integrity only; execution audit below is separately required'},
          'tabicl_kernel_shap': '500-person full computation pending; real-GPU sequential-versus-shard check pending',
          'other_model_importance': 'Seven completed model receipts; see artifacts/importance/final/h6',
          'metrics_final': 'complete: artifacts/parallel_evaluation/final-complete.json',
          'metrics_any': 'complete: artifacts/parallel_evaluation/any-complete.json',
          'imputation_comparison': 'complete: artifacts/tables/S12_imputation_comparison_h6.csv',
          'supplementary': 'S21 complete; S25/S26 and complete-source manifest pending',
          'unc': 'Cross-device TabICL strict baseline remains under investigation; remote formal SHAP held.',
          'historical_failure': 'artifacts/history/run_status_before_manual_resume_20261109.json',
          'source_status_path': 'artifacts/source_status.json'}
audit_path = ARTIFACTS / 'tabicl_permutation_execution_audit.json'
shap_stamp = shap_stage.stamp()
shap_complete = []
for i in range(importance.KS_EXPLAIN):
    path = shap_stage.WORK / f'subject_{i:03d}.npz'
    if path.exists() and verified(path.with_suffix('.done.json'), shap_stamp | {'indices': [i]}, (path,)):
        shap_complete.append(i)
status['tabicl_kernel_shap'] = {
    'sequential_isolated_control_verified': shap_stage.check_control(),
    'completed_subjects': shap_complete, 'completed_count': len(shap_complete),
    'required_subjects': importance.KS_EXPLAIN,
    'scope': 'Control participants are separate from formal subject receipts.'}
inventory_path = ARTIFACTS / 'tabicl_shap_run_inventory.json'
remote_complete, remote_issues = set(), []
if inventory_path.exists():
    inventory = json.loads(inventory_path.read_text(encoding='utf-8'))
    if (inventory.get('core_stamp') == shap_stamp
            and inventory.get('acceptance_sha256') == sha256(ARTIFACTS / 'tabicl_cross_device_acceptance.json')):
        for entry in inventory.get('runs', []):
            report_path = Path(entry.get('validation_report', ''))
            if not report_path.is_file():
                continue
            if sha256(report_path) != entry.get('validation_report_sha256'):
                remote_issues.append(f'Validation report changed: {report_path}')
                continue
            report = json.loads(report_path.read_text(encoding='utf-8'))
            if report.get('validator_sha256') != sha256(ARTIFACTS / 'tabicl_remote_shap.py'):
                remote_issues.append(f'Validation code changed: {report_path}')
                continue
            for i in report['completed_subjects']:
                item = report['evidence']['files'][str(i)]
                path = Path(item['path'])
                if (25 <= i < 500 and path.is_file() and sha256(path) == item['sha256']):
                    if i in remote_complete:
                        remote_issues.append(f'Duplicate remote subject {i}')
                    remote_complete.add(i)
                else:
                    remote_issues.append(f'Remote subject {i} changed or unavailable')
    else:
        remote_issues.append('Remote inventory frozen source or acceptance stamp is stale')
all_complete = sorted(set(shap_complete) | remote_complete)
status['tabicl_kernel_shap'].update(
    local_completed_subjects=shap_complete, local_completed_count=len(shap_complete),
    remote_received_verified_subjects=sorted(remote_complete), remote_received_verified_count=len(remote_complete),
    completed_subjects=all_complete, completed_count=len(all_complete), remote_validation_issues=remote_issues,
    scope='Formal local receipts plus received remote outputs matching independent validation reports; remote in-flight progress is separate.')
next_path = ARTIFACTS / 'tabicl_next_stages_status.json'
status['next_stages'] = json.loads(next_path.read_text(encoding='utf-8')) if next_path.exists() else None
recovery_path = ARTIFACTS / 'tabicl_local_shap_recovery.json'
if recovery_path.exists():
    status['local_shap_recovery'] = json.loads(recovery_path.read_text(encoding='utf-8'))
    status['next_stages_scope'] = 'Historical supervisor; current local progress is covered by recovery, actual processes, and subject receipts.'
if not processes and not stage_processes and set(range(25)).issubset(shap_complete):
    status['state'] = 'running'
    status['phase'] = 'awaiting_remote_results' if len(all_complete) < 500 else 'awaiting_final_aggregation'
    status['scope'] += ' No local GPU worker is required after its assigned 25 subjects finish.'
status['permutation_execution_audit'] = json.loads(audit_path.read_text(encoding='utf-8')) if audit_path.exists() else {'status': 'missing'}
acceptance_path = ARTIFACTS / 'tabicl_cross_device_acceptance.json'
if acceptance_path.exists():
    acceptance = json.loads(acceptance_path.read_text(encoding='utf-8'))
    status['unc'] = {
        'portability_decision': acceptance['status'],
        'acceptance_sha256': sha256(acceptance_path),
        'original_strict_gate': acceptance['original_strict_gate'],
        'scope': acceptance['scope'],
        'formal_status': 'UNC chat checks live scheduler state; local monitor does not establish remote startup or completion.',
    }
    dispatch_path = ARTIFACTS / 'unc/guarded_l40s_8756544/dispatch.json'
    if dispatch_path.exists():
        dispatch = json.loads(dispatch_path.read_text(encoding='utf-8'))
        status['unc']['last_scheduler_observation'] = {
            'observed_at': dispatch.get('updated'), 'queue': dispatch.get('observed_queue'),
            'remaining_estimate_at_observation': dispatch.get('remaining_estimate'),
            'scope': 'Timestamped UNC observation, not a new live scheduler check.'}
completion_path = ARTIFACTS / 'completion_audit.json'
if completion_path.exists():
    completion = json.loads(completion_path.read_text(encoding='utf-8'))
    current = (completion.get('status') == 'complete' and len(all_complete) == 500 and len(complete) == 60
               and not remote_issues
               and completion.get('implementation_sha256') == sha256(ARTIFACTS / 'audit_final_completion.py')
               and all((ARTIFACTS / name).is_file() and sha256(ARTIFACTS / name) == digest
                       for name, digest in completion.get('evidence_sha256', {}).items()))
    status['completion_audit'] = {'path': str(completion_path), 'sha256': sha256(completion_path),
                                'current_evidence_verified': current}
    if current:
        status.update(state='complete', phase='complete')
        status['other_model_importance'] = 'All eight models complete and covered by completion_audit.json'
        status['supplementary'] = '23 final tables complete; see source_status.json and completion_audit.json'
        runtime_path = ARTIFACTS / 'runtime_completion_audit.json'
        if runtime_path.exists():
            status['runtime_completion_audit'] = json.loads(runtime_path.read_text(encoding='utf-8'))
    else:
        status.update(state='needs_attention', phase='completion_evidence_changed')
save_json(ARTIFACTS / 'run_status.json', status)
print(f"{status['observed_at']}: permutation {len(complete)}/60; SHAP {len(all_complete)}/500 "
      f"(local {len(shap_complete)}, received UNC {len(remote_complete)}); {status['state']}/{status['phase']}")
