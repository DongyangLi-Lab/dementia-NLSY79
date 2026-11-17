"""NLSY79 respondent survival at the observed 2020 outcome interview."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

POLICY = 'nlsy79_endpoint_survival_v1'
RNI = dict(zip(
    list(range(1980, 1995)) + list(range(1996, 2023, 2)),
    ['R0406310','R0618810','R0898510','R1144710','R1519910','R1890300','R2257400',
     'R2444600','R2870100','R3073900','R3400500','R3655900','R4006400','R4417500',
     'R5080500','R5165800','R6478500','R7006300','R7703500','R8495900','T0987500',
     'T2209800','T3107600','T4112100','T5022300','T5770600','T8218500','T8787600','T9299500']))


def qualify_nlsy_pairs(pairs, status):
    """Death codes take precedence over an observed score, in every analysis.

    Require a completed/converted respondent interview, valid interview date,
    and no noninterview disposition at the endpoint. NDI_DECEASED_FLAG has no
    public date: report it, but do not use a future death as a pre-2020 death.
    The supported outcome wave is explicitly 2020; other waves need new maps.
    """
    d = pairs.reset_index(drop=True)
    if not d.outcome_survey_year.eq(2020).all():
        raise ValueError('NLSY survival policy only supports the 2020 outcome wave')
    raw = status.copy()
    raw.index = 'NLSY79:' + pd.to_numeric(raw.R0000100).astype('int64').astype(str)
    if not raw.index.is_unique:
        raise ValueError('Duplicate NLSY respondent identifier')
    matched = d.person_id.isin(raw.index)
    s = raw.reindex(d.person_id).reset_index(drop=True)
    prior_death = s[[col for year,col in RNI.items() if year <= 2020]].isin([65, 75]).any(axis=1)
    # The official codebook records six interviewed respondents with invalid-skip
    # mode (-3). Together, RNI=-4, a valid matching date and observed cognition
    # confirm those interviews. Sampling weights do not establish vital status.
    mode_ok = s.T8780600.isin([71, 72, 74, 75, -3])
    date_ok = (s.T8330001.between(1, 12) & s.T8330002.between(2020, 2022)
               & s.T8330001.eq(d.outcome_interview_month)
               & s.T8330002.eq(d.outcome_interview_year))
    # RNI is skipped for interviewed respondents; -4 alone is not evidence of life.
    interviewed = s.T8787600.eq(-4) & mode_ok & date_ok & d.outcome_cognition27.between(0, 27)
    result = pd.DataFrame({
        'source_matched': matched, 'death_reported_by_2020': prior_death,
        'endpoint_interview_confirmed': interviewed,
        'latest_deceased_flag': s.N0022000.isin([1, 2]),
        'death_reported_2022_only': s.T9299500.isin([65, 75]) & ~prior_death,
    })
    result['survival_eligible'] = matched & interviewed & ~prior_death
    result['reason'] = np.select([~matched, prior_death, ~interviewed],
                                ['missing_source', 'death_by_endpoint', 'unconfirmed_endpoint_interview'], default='eligible')
    result.index = pairs.index
    return result


def verify_nlsy_certificate(specification, root, csv_sha256):
    policy = specification.get('nlsy_survival_policy', {})
    if policy.get('version') != POLICY:
        raise ValueError('Missing or unsupported NLSY survival policy')
    payload = (Path(root)/policy['certificate']).read_bytes()
    if hashlib.sha256(payload).hexdigest() != policy['certificate_sha256']:
        raise ValueError('NLSY survival certificate checksum mismatch')
    certificate = json.loads(payload)
    if (certificate.get('policy') != POLICY or certificate.get('status') != 'pass'
            or certificate['stage2_nlsy_csv_sha256'] != csv_sha256
            or certificate['stage2']['excluded'] != 0 or len(certificate['branches']) != 8
            or any(b['excluded'] for b in certificate['branches'])):
        raise ValueError('NLSY data do not have a passing survival audit')
    return certificate
