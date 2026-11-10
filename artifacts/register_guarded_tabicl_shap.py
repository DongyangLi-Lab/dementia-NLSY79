"""Independently validate downloaded formal UNC shards and update the inventory."""
import json
import os
from pathlib import Path
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from artifact_store import save_json
from train.cohorts import ARTIFACTS, sha256
from artifacts.tabicl_remote_shap import validate_run
from artifacts.tabicl_kernel_shap_sharded import stamp

def main():
    inventory_path = ARTIFACTS / 'tabicl_shap_run_inventory.json'
    original_digest = sha256(inventory_path)
    inventory = json.loads(inventory_path.read_text(encoding='utf-8'))
    if inventory['core_stamp'] != stamp():
        raise ValueError('Inventory core sources changed')
    if inventory['acceptance_sha256'] != sha256(ARTIFACTS / 'tabicl_cross_device_acceptance.json'):
        raise ValueError('Acceptance decision changed')
    worker = Path(inventory['frozen_worker']['path'])
    if sha256(worker) != inventory['frozen_worker']['sha256']:
        raise ValueError('Frozen worker changed')
    validator_digest = sha256(ARTIFACTS / 'tabicl_remote_shap.py')
    runs = {r['directory']: r for r in inventory['runs']}
    seen, counts, pending = set(), {}, {}
    for folder in sorted(worker.parent.glob('shard-*-job-*')):
        try:
            records, evidence = validate_run(folder, worker)
        except (OSError, ValueError, KeyError) as exc:
            pending[folder.name] = repr(exc)
            # Retain historical entries; live counter independently rejects stale files.
            continue
        if seen.intersection(records):
            raise ValueError(f'Duplicate remote subjects in {folder}')
        seen.update(records)
        if not records:
            continue
        report_path = folder / 'local_validation.json'
        report = {'status': 'validated_production' if len(records) == 25 else 'validated_partial_production',
                  'completed_subjects': sorted(records), 'evidence': evidence,
                  'validator_sha256': validator_digest}
        save_json(report_path, report)
        runs[str(folder)] = {'directory': str(folder), 'worker_source': str(worker),
            'worker_sha256': sha256(worker), 'validation_report': str(report_path),
            'validation_report_sha256': sha256(report_path)}
        counts[folder.name] = len(records)
    if sha256(inventory_path) != original_digest:
        raise ValueError('Inventory changed during validation')
    inventory['runs'] = list(runs.values())
    save_json(inventory_path, inventory)
    print(json.dumps({'newly_verified_snapshot_counts': counts, 'verified_subjects': len(seen),
                      'pending_or_invalid_downloads': pending}), flush=True)

if __name__ == '__main__':
    main()
