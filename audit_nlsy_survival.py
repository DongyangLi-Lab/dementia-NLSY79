"""Audit NLSY79 endpoint survival with official RNI, NDI, and interview fields."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from audit_survival import digest
from part2_models.nlsy_survival import POLICY, RNI, qualify_nlsy_pairs


def summarize(frame, status):
    ledger = qualify_nlsy_pairs(frame, status)
    return {'rows': len(frame), 'people': int(frame.person_id.nunique()),
            'excluded': int((~ledger.survival_eligible).sum()),
            'retained': int(ledger.survival_eligible.sum()),
            'reasons': {str(k):int(v) for k,v in ledger.reason.value_counts().items()},
            'death_reported_by_2020': int(ledger.death_reported_by_2020.sum()),
            'latest_deceased_flag': int(ledger.latest_deceased_flag.sum()),
            'death_reported_2022_only': int(ledger.death_reported_2022_only.sum()),
            'retained_2022_death_with_observed_2020_endpoint': int((ledger.survival_eligible & ledger.death_reported_2022_only).sum())}


def audit(stage1, stage2, status_path, provenance_path, output):
    provenance = json.loads(provenance_path.read_text())
    if digest(status_path) != provenance['extract_sha256']:
        raise ValueError('NLSY survival extract checksum mismatch')
    if provenance['rni_by_year'] != {str(y):c for y,c in RNI.items()}:
        raise ValueError('RNI field mapping differs from the reviewed codebook')
    status = pd.read_csv(status_path)
    if len(status) != 12686 or not status.R0000100.is_unique:
        raise ValueError('Incomplete or duplicated NLSY source respondents')
    report = {'policy': POLICY, 'status':'pass', 'source':provenance,
              'stage1_archive_sha256':digest(stage1), 'stage2_archive_sha256':digest(stage2),
              'rule':'Exclude any RNI=65/75 through the 2020 outcome wave; require a confirmed respondent interview at the observed outcome date. Later reported death does not invalidate an earlier completed endpoint.',
              'limitation':'The public cross-round NDI deceased flag has no date. It is cross-checked, never treated as death before 2020 solely because it is positive. Survival refers to the observed endpoint interview, not December 31 or the end of all future follow-up.',
              'source_population': {'people':len(status),
                 'rni_deaths_2020':int(status.T8787600.isin([65,75]).sum()),
                 'any_rni_death_through_2020':int(status[[c for y,c in RNI.items() if y<=2020]].isin([65,75]).any(axis=1).sum()),
                 'latest_deceased_flag':int(status.N0022000.isin([1,2]).sum())},
              'branches':[]}
    prefix = 'data/processed/hrs_complete_v3/'
    with ZipFile(stage1) as archive:
        names = sorted(n for n in archive.namelist() if n.startswith((prefix+'nlsy_analysis_',prefix+'nlsy_external_full_')) and n.endswith('.csv.gz'))
        if len(names)!=8:
            raise ValueError('Expected six NLSY branches and two broader external-validation tables')
        for name in names:
            raw = gzip.decompress(archive.read(name))
            frame = pd.read_csv(io.BytesIO(raw))
            if not frame.pair_id.is_unique or not frame.person_id.is_unique:
                raise ValueError('Duplicate NLSY pair or respondent')
            report['branches'].append(dict(table=Path(name).name,csv_sha256=hashlib.sha256(raw).hexdigest(),**summarize(frame,status)))
        primary = pd.read_csv(io.BytesIO(gzip.decompress(archive.read(prefix+'nlsy_analysis_strict27_matched_age_actual4_16.csv.gz'))))
    with ZipFile(stage2) as archive:
        raw = gzip.decompress(archive.read('NLSY79.csv.gz'))
        report['stage2_nlsy_csv_sha256'] = hashlib.sha256(raw).hexdigest()
        frame = pd.read_csv(io.BytesIO(raw))
        pd.testing.assert_frame_equal(frame[['pair_id','outcome_cognition27']],primary[['pair_id','outcome_cognition27']])
        report['stage2'] = summarize(frame,status)
        report['stage2']['analysis_roles'] = {str(k):int(v) for k,v in frame.analysis_role.value_counts().items()}
        matched = status.set_index('R0000100').reindex(frame.person_id.str.split(':').str[1].astype(int))
        report['stage2']['interview_mode_invalid_skip_with_date_and_observed_score'] = int(matched.T8780600.eq(-3).sum())
    if report['stage2']['excluded'] or any(x['excluded'] for x in report['branches']):
        report['status']='fail'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_bytes((json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps({k:report[k] for k in ['status','source_population','stage2']},indent=2))
    if report['status']!='pass':
        raise ValueError('Survival-ineligible NLSY pairs require revised data before modeling')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['stage1','stage2','status','provenance','output']:
        parser.add_argument('--'+name,required=True,type=Path)
    args=parser.parse_args()
    audit(args.stage1,args.stage2,args.status,args.provenance,args.output)
