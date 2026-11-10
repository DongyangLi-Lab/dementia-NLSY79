"""Validate guarded UNC SHAP outputs without relabelling them as local outputs."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.stats import spearmanr
from artifacts import tabicl_kernel_shap_sharded as local
from artifacts.tabicl_shap_aggregation import solve_subjects
from train.cohorts import ARTIFACTS, sha256

DECISION = ARTIFACTS / 'tabicl_cross_device_acceptance.json'
INPUT_MANIFEST = ARTIFACTS / 'unc/shap_cpu_controls_private/20261110_013329/manifest.json'
IMPACT = ARTIFACTS / 'unc/shap_impact_8754716/impact_8754716.json'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def control_metrics(reference, actual, coalitions, limits):
    """Independently recompute the frozen two-person engineering guardrails."""
    require(actual['p0'].shape == (10,) and actual['p1'].shape == (500,)
            and actual['predictions'].shape == (2, 800, 10), 'Control shape mismatch')
    records = lambda data: [dict(p0=data['p0'], p1=data['p1'],
                                predictions=data['predictions'][i:i + 1]) for i in (0, 1)]
    a = solve_subjects(coalitions, [0, 1], records(reference))
    b = solve_subjects(coalitions, [0, 1], records(actual))
    fields = [('p0', None), ('p1', None), ('predictions', 0), ('predictions', 1)]
    differences = [float(np.max(np.abs((actual[key] if i is None else actual[key][i])
                                     - (reference[key] if i is None else reference[key][i]))))
                   for key, i in fields]
    require(max(differences) <= limits['shap_probability_max_abs'], 'Control probability guard failed')
    individual = []
    for i in (0, 1):
        top = np.argsort(-np.abs(a[i]), kind='stable')[:10]
        item = {'max_abs': float(np.max(np.abs(a[i] - b[i]))),
                'relative_l1': float(np.sum(np.abs(a[i] - b[i])) / np.sum(np.abs(a[i]))),
                'rho': float(spearmanr(a[i], b[i]).statistic),
                'direction_agreement': int(np.sum(np.sign(a[i, top]) == np.sign(b[i, top])))}
        require(item['max_abs'] <= limits['individual_phi_max_abs_log_odds']
                and item['relative_l1'] <= limits['individual_phi_relative_L1_max']
                and item['rho'] >= limits['individual_signed_spearman_min']
                and item['direction_agreement'] == limits['local_top10_direction_agreement_required'],
                f'SHAP control guard failed for subject {i}')
        individual.append(item)
    aa, bb = np.abs(a).mean(axis=0), np.abs(b).mean(axis=0)
    rho = float(spearmanr(aa, bb).statistic)
    overlap = len(set(np.argsort(-aa)[:10]) & set(np.argsort(-bb)[:10]))
    require(rho >= limits['mean_abs_spearman_min'] and overlap >= limits['mean_abs_top10_overlap_min'],
            'Mean absolute SHAP control guard failed')
    return {'probability_max_abs': differences, 'individual': individual,
            'mean_abs_rho': rho, 'top10_overlap': overlap}


def check_route(route, input_case):
    require(route['input_sha256'] == input_case['input']['sha256']
            and route['cache_file_sha256'] == input_case['file_sha256']
            and route['transform_hashes'] == {k: v['sha256'] for k, v in input_case['transforms'].items()},
            'Subject input provenance mismatch')
    require(route['calls'] == 24, 'Wrong number of candidate projections')
    for key, expected in {'tf_col': 112, 'tf_row': 16330, 'tf_icl': 22}.items():
        batches, offloads = route['batch'][key], route['offload'][key]
        require(batches and all(n == expected for n in batches), 'Unexpected batch geometry')
        require(offloads and all(v is (key == 'tf_col') for v in offloads), 'Unexpected offload route')


def validate_run(folder, worker_source, *, control_only=False):
    """Return verified records and evidence; accepts partial atomic manifests."""
    folder = Path(folder).resolve()
    manifest_path = folder / 'run_manifest.json'
    manifest_sha = sha256(manifest_path)
    run = read_json(manifest_path)
    decision = read_json(DECISION)
    require(decision['status'] == 'accepted_for_guarded_production', 'No active portability acceptance')
    require(run['acceptance_sha256'] == sha256(DECISION), 'Decision hash mismatch')
    require(run['core_stamp'] == local.stamp() == decision['evidence']['core_stamp'], 'Frozen sources changed')
    require(run['input_manifest_sha256'] == sha256(INPUT_MANIFEST)
            == decision['production_input_manifest_sha256'], 'Input manifest hash mismatch')
    require(run['worker_sha256'] == sha256(Path(worker_source)), 'Unrecognized production worker')
    require(sha256(IMPACT) == decision['evidence']['impact_json_sha256'], 'Impact evidence changed')
    sources = read_json(IMPACT)['files_sha256']
    require(run['patch_library_sha256'] == sources['icl_cublaslt_split_rows.so']
            and run['diagnostic_source_sha256'] == sources['icl_shap_controls_candidate_diag.py'],
            'Unrecognized inference patch')
    require(run['gpu'] == 'NVIDIA L40S', 'Execution was not on the accepted GPU type')
    require(run['status'] in ('running', 'complete'), 'Run has no importable state')
    idx, indices = run['array_index'], run['indices']
    if control_only:
        require(idx == 0 and indices == [] and run['completed'] == {}, 'Not a control-only run')
    else:
        require(isinstance(idx, int) and 1 <= idx <= 19, 'Remote array outside ownership')
        require(indices and len(indices) == len(set(indices))
                and all(isinstance(i, int) and idx * 25 <= i < (idx + 1) * 25 for i in indices),
                'Assigned subjects outside the remote shard')
    report_path, control_path = folder / 'startup_checks.json', folder / 'startup_predictions.npz'
    require(sha256(report_path) == run['startup_report_sha256']
            and sha256(control_path) == run['startup_predictions_sha256'], 'Startup evidence hash mismatch')
    report, controls = read_json(report_path), read_npz(control_path)
    require(report['status'] == 'passed_engineering_gate'
            and np.isfinite(report['full_max_abs'])
            and report['full_max_abs'] <= decision['startup_checks']['full_cohort_probability_max_abs'],
            'Full baseline startup guard failed')
    require(local.check_control(), 'Local reference control is stale')
    design = read_npz(local.DESIGN)
    measured = control_metrics(read_npz(local.WORK / 'sequential_reference.npz'), controls,
                               design['coalitions'], decision['startup_checks'])
    for key in ('probability_max_abs', 'mean_abs_rho', 'top10_overlap'):
        require(np.allclose(measured[key], report[key], rtol=1e-9, atol=1e-12), 'Control report inconsistent')
    require(len(report['individual']) == 2, 'Missing individual control report')
    for measured_item, reported_item in zip(measured['individual'], report['individual']):
        for key, value in measured_item.items():
            require(np.isclose(value, reported_item[key], rtol=1e-9, atol=1e-12), 'Individual report inconsistent')
    input_cases = read_json(INPUT_MANIFEST)['cases']
    for i in (0, 1):
        check_route(report['routes'][f'subject_{i:03d}'], input_cases[f'subject_{i:03d}'])
    completed = run['completed']
    require(all(str(i) in {str(j) for j in indices} for i in completed), 'Unexpected completed subject')
    if run['status'] == 'complete':
        require(set(completed) == {str(i) for i in indices}, 'Complete run is missing subjects')
    records, files = {}, {}
    for key, entry in completed.items():
        i = int(key)
        require(entry['file'] == f'subject_{i:03d}.npz', 'Unexpected output filename')
        path = folder / entry['file']
        require(sha256(path) == entry['sha256'], f'Output hash mismatch for subject {i}')
        record = read_npz(path)
        require(np.array_equal(record['p0'], controls['p0']) and np.array_equal(record['p1'], controls['p1']),
                'Subject warmups do not match its recorded execution')
        # This also validates shape, dtype, finiteness, and probability bounds.
        solve_subjects(design['coalitions'], [i], [record])
        check_route(entry['route'], input_cases[f'subject_{i:03d}'])
        records[i], files[str(i)] = record, {'path': str(path), 'sha256': entry['sha256']}
    require(sha256(manifest_path) == manifest_sha, 'Manifest changed while being read')
    return records, {'manifest': str(manifest_path), 'manifest_sha256': manifest_sha,
                     'control_only': control_only,
                     'job_id': run['job_id'], 'worker_sha256': run['worker_sha256'],
                     'startup_metrics': measured, 'files': files, 'original_strict_gate': run['strict_gate']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_directory', type=Path)
    parser.add_argument('--worker-source', type=Path, required=True)
    parser.add_argument('--control-only', action='store_true')
    args = parser.parse_args()
    records, evidence = validate_run(args.run_directory, args.worker_source, control_only=args.control_only)
    print(json.dumps({'validated_subjects': sorted(records), 'evidence': evidence}, indent=2))
