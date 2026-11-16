"""Audit frozen HRS analysis pairs against authorized local HRS Tracker data."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from part2_models.survival import POLICY, WAVES, qualify_pairs


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def summarize(frame, tracker):
    ledger = qualify_pairs(frame, tracker)
    return {
        'rows': len(frame), 'people': int(frame.person_id.nunique()),
        'retained': int(ledger.survival_eligible.sum()),
        'excluded': int((~ledger.survival_eligible).sum()),
        'reasons': {str(k): int(v) for k, v in ledger.reason.value_counts().items()},
        'endpoint_wave_counts': {str(k): int(v) for k, v in frame.outcome_survey_year.value_counts().sort_index().items()},
    }


def audit(stage1, stage2, tracker_path, output):
    columns = ['HHID', 'PN', 'KNOWNDECEASEDYR', 'KNOWNDECEASEDMO'] + [p+'ALIVE' for p in WAVES.values()]
    tracker = pd.read_stata(tracker_path, columns=columns, convert_categoricals=False)
    report = {'policy': POLICY, 'status': 'pass', 'tracker_sha256': digest(tracker_path),
              'stage1_archive_sha256': digest(stage1), 'stage2_archive_sha256': digest(stage2),
              'rule': 'Require endpoint ALIVE=1; exclude any death flag 5/6 within the pair window or death known before its endpoint/interview. Retain eligible earlier pairs despite later death.',
              'scope': 'All six HRS analysis branches, all-pair and first-pair tables, including official27 sensitivity. NLSY79 unchanged; no independent NLSY mortality registry audit.',
              'branches': []}
    prefix = 'data/processed/hrs_complete_v3/'
    primary = prefix+'hrs_analysis_allpairs_strict27_matched_age_actual4_16.csv.gz'
    with ZipFile(stage1) as source:
        names = sorted(n for n in source.namelist() if n.startswith(prefix+'hrs_analysis_') and n.endswith('.csv.gz'))
        if len(names) != 12:
            raise ValueError('Expected six all-pair and six first-pair analysis tables')
        for name in names:
            raw = gzip.decompress(source.read(name))
            frame = pd.read_csv(io.BytesIO(raw))
            if not frame.pair_id.is_unique:
                raise ValueError('Duplicate pair in '+name)
            report['branches'].append(dict(table=Path(name).name, csv_sha256=hashlib.sha256(raw).hexdigest(), **summarize(frame, tracker)))
        original = pd.read_csv(io.BytesIO(gzip.decompress(source.read(primary))))
    with ZipFile(stage2) as source:
        raw = gzip.decompress(source.read('HRS.csv.gz'))
        report['stage2_hrs_csv_sha256'] = hashlib.sha256(raw).hexdigest()
        hrs = pd.read_csv(io.BytesIO(raw))
        pd.testing.assert_frame_equal(hrs[['pair_id', 'outcome_cognition27']], original[['pair_id', 'outcome_cognition27']])
        report['stage2'] = summarize(hrs, tracker)
        nlsy_raw = gzip.decompress(source.read('NLSY79.csv.gz'))
        nlsy = pd.read_csv(io.BytesIO(nlsy_raw))
        report['nlsy79'] = {'rows': len(nlsy), 'csv_sha256': hashlib.sha256(nlsy_raw).hexdigest(),
                            'observed_outcomes': int(nlsy.outcome_cognition27.notna().sum()),
                            'independent_mortality_audit': False}
    if report['stage2']['excluded'] or any(b['excluded'] for b in report['branches']):
        report['status'] = 'fail'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes((json.dumps(report, indent=2)+'\n').encode('utf-8'))
    print(json.dumps({'status': report['status'], 'tables': len(report['branches']), 'HRS': report['stage2'], 'report': str(output)}, indent=2))
    if report['status'] != 'pass':
        raise ValueError('Frozen analysis contains survival-ineligible pairs; rebuild before training')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage1', required=True, type=Path)
    parser.add_argument('--stage2', required=True, type=Path)
    parser.add_argument('--tracker', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    audit(args.stage1, args.stage2, args.tracker, args.output)
