"""Engineering fixtures only; these observations are not study data."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from part2_models.data import TARGET, pair_weights, sha256
from part2_models.estimators import FittedRegressor
from part2_models.evaluation import paired_bootstrap
from part2_models.store import finish_stage, load_model, save_model, start_stage, verify_stage, write_json
from part2_models.workflow import evaluate


def fixture():
    return pd.DataFrame({'person_id': ['p1', 'p1', 'p2', 'p3', 'p4', 'p5'],
                         'household_id': ['h1', 'h1', 'h2', 'h2', 'h3', 'h4'],
                         'x': [1., 2., np.nan, 4., 5., 6.],
                         'race3__1': [1., 0., np.nan, 0., 1., 0.],
                         'race3__2': [0., 1., np.nan, 0., 0., 1.],
                         'race3__3': [0., 0., np.nan, 1., 0., 0.],
                         'baseline_cognition27': [15., 15., 20., 22., 18., 19.],
                         'followup_years': [4., 6., 8., 10., 12., 14.]})


FEATURES = ['x', 'race3__1', 'race3__2', 'race3__3', 'baseline_cognition27', 'followup_years']


class PipelineTests(unittest.TestCase):
    def test_equal_total_person_weights(self):
        frame = fixture()
        frame['weight'] = pair_weights(frame)
        totals = frame.groupby('person_id').weight.sum()
        np.testing.assert_allclose(totals, totals.iloc[0])

    def test_fit_only_imputation_and_feature_schema(self):
        frame = fixture()
        model = FittedRegressor('ridge', FEATURES, {'alpha': 1}).fit(frame, [14, 15, 19, 20, 17, 18])
        before = model.preprocessor.named_transformers_['numeric'].named_steps['impute'].statistics_.copy()
        unseen = frame.iloc[[0]].copy()
        unseen['x'] = 1000000
        model.predict(unseen)
        np.testing.assert_array_equal(before, model.preprocessor.named_transformers_['numeric'].named_steps['impute'].statistics_)
        self.assertEqual(before[0], 4.)
        with self.assertRaises(ValueError):
            model.predict(unseen.drop(columns='x'))

    def test_categorical_missing_and_unknown(self):
        frame = fixture()
        frame['retired_status_raw'] = [0, 0, 1, 1, np.nan, 1]
        model = FittedRegressor('ridge', FEATURES+['retired_status_raw'], {'alpha': 1}).fit(frame, [14, 15, 19, 20, 17, 18])
        new = frame.iloc[[0]].copy()
        new['retired_status_raw'] = 999
        new['x'] = np.nan
        self.assertTrue(np.isfinite(model.predict(new)).all())
        self.assertEqual(model.frame(frame).loc[2, 'race3'], 'MISSING')
        new['race3__1'] = np.nan
        with self.assertRaises(ValueError):
            model.predict(new)

    def test_trajectory_zero_time(self):
        frame = fixture()
        model = FittedRegressor('trajectory', FEATURES, {'alpha': 1}).fit(frame, [14, 15, 19, 20, 17, 18])
        frame['followup_years'] = 0
        np.testing.assert_array_equal(model.predict(frame), frame.baseline_cognition27)

    def test_model_replay_and_corruption(self):
        frame = fixture()
        model = FittedRegressor('ridge', FEATURES, {'alpha': 1}).fit(frame, [14, 15, 19, 20, 17, 18])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'saved'
            save_model(model, path, {'purpose': 'unit_test'}, frame)
            loaded, _ = load_model(path)
            np.testing.assert_array_equal(model.predict(frame), loaded.predict(frame))
            with (path/'pipeline.joblib').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaises(ValueError):
                load_model(path)

    def test_stage_reuse_and_contract_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'stage'
            self.assertTrue(start_stage(path, {'stage': 'fixture', 'seed': 42}))
            (path/'result.txt').write_text('test')
            finish_stage(path)
            self.assertFalse(start_stage(path, {'stage': 'fixture', 'seed': 42}))
            with self.assertRaises(ValueError):
                start_stage(path, {'stage': 'fixture', 'seed': 43})
            (path/'result.txt').write_text('changed')
            with self.assertRaises(ValueError):
                verify_stage(path)

    def test_paired_bootstrap_identical_predictions(self):
        result = paired_bootstrap([1, 2, 3, 4], [2, 3, 2, 5], [2, 3, 2, 5], ['a', 'a', 'b', 'c'], 30)
        self.assertEqual(result['RMSE_A_minus_B'], 0)
        self.assertEqual(result['CI95_low'], 0)
        self.assertEqual(result['CI95_high'], 0)
        self.assertEqual(result['bootstrap_households'], 3)

    def test_evaluation_join_and_pilot_guard(self):
        # Exercise the final evaluation path only on labeled synthetic fixtures.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root/'fixture.zip'
            data.write_bytes(b'synthetic test data placeholder')
            transfer = root/'transfer'
            target = root/'target'
            config = {'pilot_households_per_fold': None, 'seed': 42, 'bootstrap_repeats': 20,
                      'target_grids': {'ridge': {}}, 'primary_target_learner': 'ridge', 'primary_increment': 'B_change'}
            frame = pd.DataFrame({'pair_id': ['p1', 'p2', 'p3'], 'person_id': ['p1', 'p2', 'p3'],
                                  'household_id': ['h1', 'h1', 'h2'], 'followup_band': ['4_to_lt8']*3,
                                  'analysis_role': ['locked_test']*3, TARGET: [14., 18., 20.]})
            start_stage(transfer, {'stage': 'fixture'})
            direct = frame[['pair_id']].copy()
            for name in ['S_score', 'trajectory_endpoint', 'simple_prediction', 'persistence_prediction']:
                direct[name] = [15., 18., 19.]
            direct.to_csv(transfer/'transfer_predictions.csv.gz', index=False)
            finish_stage(transfer)
            start_stage(target, {'data_sha256': sha256(data), 'config': config,
                                 'transfer_manifest_sha256': sha256(transfer/'_SUCCESS.json')})
            predictions = frame[['pair_id']].copy()
            for variant in ['C', 'A', 'B_score', 'B_change']:
                predictions['ridge_'+variant] = [15., 18., 19.]
            predictions.to_csv(target/'locked_predictions.csv.gz', index=False)
            finish_stage(target)
            with patch('part2_models.workflow.load_data', return_value=({'NLSY79': frame}, {})):
                evaluate(data, transfer, target, config, root/'evaluation')
            report = json.loads((root/'evaluation/evaluation.json').read_text())
            self.assertEqual(report['locked_people'], 3)
            self.assertEqual(report['locked_households'], 2)
            verify_stage(root/'evaluation')
            pilot = dict(config, pilot_households_per_fold=1)
            pilot_target = root/'pilot_target'
            start_stage(pilot_target, {'data_sha256': sha256(data), 'config': pilot,
                                      'transfer_manifest_sha256': sha256(transfer/'_SUCCESS.json')})
            finish_stage(pilot_target)
            with self.assertRaisesRegex(ValueError, 'Pilot runs cannot'):
                evaluate(data, transfer, pilot_target, pilot, root/'forbidden_evaluation')
            self.assertFalse((root/'forbidden_evaluation').exists())


if __name__ == '__main__':
    unittest.main()
