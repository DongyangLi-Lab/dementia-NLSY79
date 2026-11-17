"""Engineering fixtures only; not scientific observations."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from part2_models.nlsy_survival import POLICY, RNI, qualify_nlsy_pairs, verify_nlsy_certificate


def fixture():
    pairs=pd.DataFrame({'person_id':['NLSY79:1'],'outcome_survey_year':[2020],
                        'outcome_interview_year':[2021],'outcome_interview_month':[9],
                        'outcome_cognition27':[24.]})
    status=pd.DataFrame({'R0000100':[1], 'N0022000':[-4], 'T8780600':[72],
                         'T8787500':[150000.], 'T8330001':[9], 'T8330002':[2021],
                         **{c:[-4] for c in RNI.values()}})
    return pairs,status


class NlsySurvivalTests(unittest.TestCase):
    def test_endpoint_or_prior_death_overrides_observed_score(self):
        for year in [2004,2018,2020]:
            for code in [65,75]:
                pairs,status=fixture();status[RNI[year]]=code
                self.assertFalse(qualify_nlsy_pairs(pairs,status).survival_eligible.iloc[0])

    def test_later_death_retains_observed_endpoint(self):
        pairs,status=fixture();status['N0022000']=2;status[RNI[2022]]=65
        result=qualify_nlsy_pairs(pairs,status)
        self.assertTrue(result.survival_eligible.iloc[0])
        self.assertTrue(result.death_reported_2022_only.iloc[0])

    def test_undated_latest_death_not_backdated(self):
        pairs,status=fixture();status['N0022000']=1
        self.assertTrue(qualify_nlsy_pairs(pairs,status).survival_eligible.iloc[0])

    def test_observed_score_alone_cannot_override_noninterview(self):
        for code in [61,64,66,-5]:
            pairs,status=fixture();status[RNI[2020]]=code
            self.assertFalse(qualify_nlsy_pairs(pairs,status).survival_eligible.iloc[0])

    def test_missing_mode_requires_documented_interview_evidence(self):
        pairs,status=fixture();status['T8780600']=-3
        self.assertTrue(qualify_nlsy_pairs(pairs,status).survival_eligible.iloc[0])
        status['T8787500']=0
        self.assertTrue(qualify_nlsy_pairs(pairs,status).survival_eligible.iloc[0])
        pairs['outcome_cognition27']=float('nan')
        self.assertFalse(qualify_nlsy_pairs(pairs,status).survival_eligible.iloc[0])

    def test_dates_must_match(self):
        pairs,status=fixture();status['T8330001']=8
        self.assertFalse(qualify_nlsy_pairs(pairs,status).survival_eligible.iloc[0])

    def test_missing_duplicate_and_unsupported_source(self):
        pairs,status=fixture()
        with self.assertRaises(ValueError):qualify_nlsy_pairs(pairs,pd.concat([status,status]))
        pairs.person_id='NLSY79:2'
        self.assertFalse(qualify_nlsy_pairs(pairs,status).survival_eligible.iloc[0])
        pairs.outcome_survey_year=2022
        with self.assertRaises(ValueError):qualify_nlsy_pairs(pairs,status)

    def test_certificate_binds_exact_table(self):
        cert={'policy':POLICY,'status':'pass','stage2_nlsy_csv_sha256':'table',
              'stage2':{'excluded':0},'branches':[{'excluded':0} for _ in range(8)]}
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'audit.json';p.write_text(json.dumps(cert))
            spec={'nlsy_survival_policy':{'version':POLICY,'certificate':p.name,
                  'certificate_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}}
            verify_nlsy_certificate(spec,temp,'table')
            with self.assertRaises(ValueError):verify_nlsy_certificate(spec,temp,'changed')
            p.write_text('{}')
            with self.assertRaises(ValueError):verify_nlsy_certificate(spec,temp,'table')
