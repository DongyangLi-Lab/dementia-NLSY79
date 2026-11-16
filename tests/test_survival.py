"""Engineering fixtures for death precedence, pair endpoints, and audit binding."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from part2_models.survival import POLICY, WAVES, qualify_pairs, verify_certificate


def fixture():
    pairs = pd.DataFrame({'person_id': ['HRS:1001'] * 2, 'survey_year': [2004, 2004],
                          'outcome_survey_year': [2010, 2016],
                          'outcome_interview_year': [2010, 2016],
                          'outcome_interview_month': [6, 6], 'outcome_cognition27': [25, 24]})
    tracker = pd.DataFrame({'HHID': ['000001'], 'PN': ['001'],
                            'KNOWNDECEASEDYR': [np.nan], 'KNOWNDECEASEDMO': [np.nan],
                            **{p+'ALIVE': [1] for p in WAVES.values()}})
    return pairs, tracker


class SurvivalTests(unittest.TestCase):
    def test_later_death_preserves_earlier_pair(self):
        pairs, tracker = fixture()
        tracker['KNOWNDECEASEDYR'] = 2015
        tracker['PALIVE'] = 5
        self.assertEqual(qualify_pairs(pairs, tracker).survival_eligible.tolist(), [True, False])

    def test_death_flags_override_observed_scores(self):
        for flag in [5, 6]:
            pairs, tracker = fixture()
            tracker['KALIVE'] = flag  # Intermediate 2006 wave, endpoint code still 1.
            self.assertFalse(qualify_pairs(pairs, tracker).survival_eligible.any())

    def test_unconfirmed_alive_or_unmatched_excluded(self):
        for flag in [2, np.nan]:
            pairs, tracker = fixture()
            tracker['MALIVE'] = flag
            self.assertFalse(qualify_pairs(pairs, tracker).survival_eligible.iloc[0])
        pairs.person_id = 'HRS:9999'
        self.assertTrue(qualify_pairs(pairs, tracker).reason.eq('missing_tracker').all())

    def test_death_date_precedes_interview_not_calendar_year_end(self):
        pairs, tracker = fixture()
        tracker['KNOWNDECEASEDYR'] = 2010
        for month, expected in [(5, False), (6, True), (7, True)]:
            tracker['KNOWNDECEASEDMO'] = month
            self.assertEqual(bool(qualify_pairs(pairs, tracker).survival_eligible.iloc[0]), expected)
        # Delayed interview in a later calendar year: known earlier year is sufficient.
        pairs['outcome_interview_year'] = 2011
        tracker['KNOWNDECEASEDMO'] = np.nan
        self.assertFalse(qualify_pairs(pairs, tracker).survival_eligible.iloc[0])

    def test_invalid_waves_and_duplicate_tracker_rejected(self):
        pairs, tracker = fixture()
        with self.assertRaises(ValueError):
            qualify_pairs(pairs, pd.concat([tracker, tracker]))
        pairs['outcome_survey_year'] = 2024
        with self.assertRaises(ValueError):
            qualify_pairs(pairs, tracker)

    def test_preserves_nonconsecutive_input_index_and_lowercase_tracker(self):
        pairs, tracker = fixture()
        pairs.index = [7, 19]
        tracker.columns = tracker.columns.str.lower()
        result = qualify_pairs(pairs, tracker)
        self.assertEqual(result.index.tolist(), [7, 19])
        self.assertTrue(result.survival_eligible.all())

    def test_certificate_bound_to_table_and_content(self):
        certificate = {'policy': POLICY, 'status': 'pass', 'stage2_hrs_csv_sha256': 'tablehash',
                       'stage2': {'excluded': 0}, 'branches': [{'excluded': 0}]}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'audit.json'
            path.write_text(json.dumps(certificate))
            spec = {'survival_policy': {'version': POLICY, 'certificate': path.name,
                    'certificate_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}}
            verify_certificate(spec, temp, 'tablehash')
            with self.assertRaises(ValueError):
                verify_certificate(spec, temp, 'different_data')
            path.write_text('{}')
            with self.assertRaises(ValueError):
                verify_certificate(spec, temp, 'tablehash')


if __name__ == '__main__':
    unittest.main()
