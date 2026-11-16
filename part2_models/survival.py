"""Pair-specific HRS survival eligibility; never a predictor or outcome recoding."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

POLICY = 'hrs_pair_endpoint_survival_v1'
WAVES = dict(zip(range(2000, 2024, 2), 'GHJKLMNOPQRS'))


def qualify_pairs(pairs, tracker):
    """Return an audit ledger in input row order, using official Tracker codes.

    ALIVE=1 confirms survival; 2 is only presumed alive, 5/6 denote death.
    Death overrides an observed score. Later death does not invalidate earlier
    pairs. In a shared calendar year, only a known earlier death month overrides
    the actual interview; equal months cannot resolve day-level ordering.
    """
    d = pairs.reset_index(drop=True)
    t = tracker.copy()
    t.columns = t.columns.str.upper()
    identifiers = pd.to_numeric(t.HHID) * 1000 + pd.to_numeric(t.PN)
    t.index = 'HRS:' + identifiers.astype('int64').astype(str)
    if not t.index.is_unique:
        raise ValueError('Duplicate Tracker person identifier')
    baseline = pd.to_numeric(d['baseline_survey_year'] if 'baseline_survey_year' in d else d['survey_year'])
    endpoint = pd.to_numeric(d.outcome_survey_year)
    if not (baseline.isin(WAVES) & endpoint.isin(WAVES) & endpoint.gt(baseline)).all():
        raise ValueError('Unsupported or invalid pair wave interval')
    matched = d.person_id.isin(t.index)
    t = t.reindex(d.person_id).reset_index(drop=True)
    confirmed = np.zeros(len(d), dtype=bool)
    death = np.zeros(len(d), dtype=bool)
    for year, prefix in WAVES.items():
        vital = pd.to_numeric(t[prefix + 'ALIVE'], errors='coerce')
        confirmed |= (endpoint.eq(year) & vital.eq(1)).to_numpy()
        death |= (baseline.le(year) & endpoint.ge(year) & vital.isin([5, 6])).to_numpy()
    year = pd.to_numeric(t.KNOWNDECEASEDYR, errors='coerce')
    month = pd.to_numeric(t.KNOWNDECEASEDMO, errors='coerce')
    iy = pd.to_numeric(d.outcome_interview_year, errors='coerce')
    im = pd.to_numeric(d.outcome_interview_month, errors='coerce')
    known_year = year.between(1900, 2100)
    prior_year = known_year & year.lt(endpoint)
    prior_interview = known_year & iy.between(1900, 2100) & (
        year.lt(iy) | (year.eq(iy) & month.between(1, 12) & im.between(1, 12) & month.lt(im)))
    result = pd.DataFrame({
        'tracker_matched': matched, 'confirmed_alive_at_endpoint': confirmed,
        'death_flag_in_pair_window': death, 'death_before_endpoint_year': prior_year,
        'death_before_interview': prior_interview,
    })
    result['survival_eligible'] = matched & confirmed & ~death & ~prior_year & ~prior_interview
    result['reason'] = np.select(
        [~matched, death | prior_year | prior_interview, ~confirmed],
        ['missing_tracker', 'death_before_endpoint', 'unconfirmed_survival'], default='eligible')
    result.index = pairs.index
    return result


def verify_certificate(specification, root, hrs_csv_sha256):
    """Require the independently audited exact HRS table before any model use."""
    policy = specification.get('survival_policy', {})
    if policy.get('version') != POLICY:
        raise ValueError('Missing or unsupported survival policy')
    path = Path(root) / policy['certificate']
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != policy['certificate_sha256']:
        raise ValueError('Survival certificate checksum mismatch')
    certificate = json.loads(payload)
    if (certificate.get('policy') != POLICY or certificate.get('status') != 'pass'
            or certificate['stage2_hrs_csv_sha256'] != hrs_csv_sha256
            or certificate['stage2']['excluded'] != 0
            or not certificate['branches']
            or any(branch['excluded'] for branch in certificate['branches'])):
        raise ValueError('HRS data do not have a passing survival audit')
    return certificate
