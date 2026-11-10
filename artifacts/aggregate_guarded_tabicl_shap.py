"""Aggregate exactly 500 verified subjects across local and guarded UNC runs."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from artifact_store import save_json, save_receipt, verified, write_file
from artifacts import tabicl_kernel_shap_sharded as local
from artifacts.tabicl_remote_shap import DECISION, read_json, read_npz, require, validate_run
from artifacts.tabicl_shap_aggregation import solve_subjects
from train.cohorts import ARTIFACTS, sha256

INVENTORY = ARTIFACTS / 'tabicl_shap_run_inventory.json'
SCRIPT = Path(__file__).resolve()


def aggregate():
    inventory = read_json(INVENTORY)
    inventory_sha = sha256(INVENTORY)
    require(inventory['acceptance_sha256'] == sha256(DECISION), 'Inventory acceptance is stale')
    allocation = read_json(ARTIFACTS / 'shap_dispatch_ownership.json')
    require(allocation['local']['inclusive_indices'] == [0, 24]
            and allocation['unc']['inclusive_indices'] == [25, 499], 'Ownership changed')
    require(local.check_control(), 'Local execution control is stale')
    stamp = local.stamp()
    require(inventory['core_stamp'] == stamp, 'Inventory frozen sources changed')
    records, sources, runs = {}, {}, []
    reference = read_npz(local.WORK / 'sequential_reference.npz')
    for i in range(25):
        path = local.WORK / f'subject_{i:03d}.npz'
        receipt = path.with_suffix('.done.json')
        if verified(receipt, stamp | {'indices': [i]}, (path,)):
            record = read_npz(path)
            require(all(np.allclose(record[k], reference[k], rtol=1e-5, atol=1e-7)
                        for k in ('p0', 'p1')), 'Local warmup control mismatch')
            records[i] = record
            sources[str(i)] = {'kind': 'local', 'path': str(path), 'sha256': sha256(path),
                               'receipt_sha256': sha256(receipt)}
    for entry in inventory['runs']:
        require(sha256(Path(entry['worker_source'])) == entry['worker_sha256'], 'Worker source snapshot changed')
        remote, evidence = validate_run(entry['directory'], entry['worker_source'])
        for i, record in remote.items():
            require(i not in records, f'Duplicate subject {i} in received runs')
            records[i] = record
            sources[str(i)] = {'kind': 'unc', **evidence['files'][str(i)],
                               'job_id': evidence['job_id'], 'manifest_sha256': evidence['manifest_sha256']}
        runs.append(evidence)
    missing = sorted(set(range(500)) - records.keys())
    require(not missing, f'Incomplete SHAP: {len(records)}/500 verified; missing {missing}')
    design = read_npz(local.DESIGN)
    phi = solve_subjects(design['coalitions'], list(range(500)), [records[i] for i in range(500)])
    require(phi.shape == (500, 60), 'Unexpected final SHAP shape')
    require(sha256(INVENTORY) == inventory_sha and local.stamp() == stamp, 'Sources changed during aggregation')
    output = local.WORK / 'aggregated_cross_device.npz'
    def write(path):
        with path.open('wb') as stream:
            np.savez_compressed(stream, groups=design['groups'], values=phi)
    write_file(output, write)
    provenance = local.WORK / 'aggregated_cross_device_sources.json'
    save_json(provenance, {'status': 'complete', 'scope': '500 subjects, 60 groups; per-execution p0/p1',
                          'acceptance_sha256': sha256(DECISION), 'inventory_sha256': inventory_sha,
                          'core_stamp': stamp, 'subjects': sources, 'remote_runs': runs,
                          'original_strict_cross_device_gate': 'failed; accepted under recorded engineering criteria'})
    implementation = {p.name: sha256(p) for p in (
        SCRIPT, SCRIPT.with_name('tabicl_remote_shap.py'), SCRIPT.with_name('tabicl_shap_aggregation.py'))}
    save_receipt(output.with_suffix('.done.json'), stamp | {'acceptance': sha256(DECISION),
                 'inventory': inventory_sha, 'implementation': implementation}, (output, provenance))
    print(f'Aggregated verified mixed-execution SHAP: {phi.shape}', flush=True)
    return output


if __name__ == '__main__':
    aggregate()
