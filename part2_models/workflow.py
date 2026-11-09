"""Source training, transfer, downstream comparisons, and locked evaluation."""
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import TARGET, load_data, pair_weights, require, select_households, sha256
from .estimators import FittedRegressor
from .evaluation import metrics, paired_bootstrap
from .store import (finish_stage, load_model, save_model, start_stage, verify_stage, write_json)


def settings(grid):
    return [dict(zip(grid, values)) for values in itertools.product(*grid.values())]


def search(frame, features, learner, grid, config, balanced):
    folds = sorted(frame.cv_fold5.unique())
    require(len(folds) >= 2, 'At least two household folds are required')
    scores, candidates, predictions = [], [], []
    for candidate, params in enumerate(settings(grid)):
        prediction = np.full(len(frame), np.nan)
        started = time.monotonic()
        for fold in folds:
            training = frame.loc[frame.cv_fold5.ne(fold)]
            mask = frame.cv_fold5.eq(fold).to_numpy()
            validation = frame.loc[mask]
            require(not set(training.household_id) & set(validation.household_id), 'Household leakage')
            model = FittedRegressor(learner, features, params, config['seed'], config['threads'])
            model.fit(training, training[TARGET], pair_weights(training) if balanced else None)
            prediction[mask] = model.predict(validation)
        require(np.isfinite(prediction).all(), 'Incomplete out-of-fold predictions')
        score = metrics(frame[TARGET], prediction, pair_weights(frame) if balanced else None)
        scores.append({'candidate': candidate, 'learner': learner, 'params': json.dumps(params, sort_keys=True),
                       **score, 'seconds': time.monotonic()-started})
        candidates.append(params)
        predictions.append(prediction)
        print(learner, candidate, 'development CV RMSE', round(score['RMSE'], 4), flush=True)
    best = min(range(len(scores)), key=lambda index: scores[index]['RMSE'])
    return candidates[best], scores, predictions[best]


def train_one(frame, features, learner, grid, config, balanced, out, label, provenance=None):
    selected, scores, prediction = search(frame, features, learner, grid, config, balanced)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(scores).to_csv(out/'search.csv', index=False)
    oof = frame[['pair_id', 'person_id', 'household_id', 'cv_fold5']].copy()
    oof['prediction'] = prediction
    oof.to_csv(out/'development_oof.csv.gz', index=False)
    model = FittedRegressor(learner, features, selected, config['seed'], config['threads'])
    weights = pair_weights(frame) if balanced else None
    model.fit(frame, frame[TARGET], weights)
    training = frame[['pair_id', 'person_id', 'household_id', 'cv_fold5']].copy()
    training['fit_weight'] = weights if balanced else 1.0
    training.to_csv(out/'training_rows.csv.gz', index=False)
    score = metrics(frame[TARGET], prediction, weights)
    save_model(model, out/'model', {**(provenance or {}), 'label': label, 'training_rows': len(frame),
               'training_people': frame.person_id.nunique(), 'selection_cv_metrics': score,
               'training_manifest_sha256': sha256(out/'training_rows.csv.gz'),
               'selection_note': 'Development estimate after hyperparameter selection; not an unbiased test estimate'},
               frame.iloc[:32])
    return model, score


def source(data, config, out):
    frames, spec = load_data(data)
    contract = {'stage': 'source', 'data_sha256': sha256(data), 'config': config}
    if not start_stage(out, contract):
        return
    frame = select_households(frames['HRS'], config.get('pilot_households_per_fold'))
    features = spec['features']['feature_sets'][config['source_feature_set']]
    if config.get('exclude_education'):
        features = [c for c in features if not c.startswith('education4__')]
    if config.get('first_pair_only'):
        frame = frame.loc[frame.is_first_eligible_pair.eq(1)].reset_index(drop=True)
    selected = []
    for learner, grid in config['endpoint_grids'].items():
        _, score = train_one(frame, features, learner, grid, config, True, out/'endpoint'/learner, learner)
        selected.append({'learner': learner, **score})
    winner = min(selected, key=lambda row: row['RMSE'])['learner']
    _, trajectory_score = train_one(frame, features, 'trajectory', config['trajectory_grid'], config, True,
                                    out/'trajectory', 'conditional_change')
    simple = ['baseline_cognition27', 'age_years', 'followup_years']
    _, simple_score = train_one(frame, simple, 'ridge', {'alpha': [1e-6]}, config, True, out/'simple', 'simple_reference')
    write_json(out/'selection.json', {'endpoint_winner': winner, 'endpoint_cv': selected,
               'trajectory_cv': trajectory_score, 'simple_cv': simple_score,
               'persistence_cv': metrics(frame[TARGET], frame.baseline_cognition27, pair_weights(frame)),
               'feature_set': config['source_feature_set'], 'features': features,
               'pilot': config.get('pilot_households_per_fold') is not None})
    finish_stage(out)


def transfer(data, source_dir, out):
    source_manifest = verify_stage(source_dir)
    require(source_manifest['contract']['data_sha256'] == sha256(data), 'Source data contract differs')
    contract = {'stage': 'transfer', 'data_sha256': sha256(data),
                'source_manifest_sha256': sha256(source_dir/'_SUCCESS.json')}
    if not start_stage(out, contract):
        return
    frames, _ = load_data(data)
    target = frames['NLSY79']
    selection = json.loads((source_dir/'selection.json').read_text())
    paths = {'S_score': source_dir/'endpoint'/selection['endpoint_winner']/'model',
             'trajectory_endpoint': source_dir/'trajectory/model', 'simple_prediction': source_dir/'simple/model'}
    predictions = target[['pair_id', 'person_id', 'household_id', 'cv_fold5', 'analysis_role', 'followup_years', 'followup_band']].copy()
    provenance = {}
    # Only predictors enter predict(); no target outcome is saved or scored here.
    for column, directory in paths.items():
        model, info = load_model(directory)
        predictions[column] = model.predict(target.loc[:, model.features])
        provenance[column] = info['pipeline_sha256']
    predictions['S_delta'] = predictions.trajectory_endpoint-target.baseline_cognition27
    predictions['endpoint_delta'] = predictions.S_score-target.baseline_cognition27
    predictions['persistence_prediction'] = target.baseline_cognition27
    predictions.to_csv(out/'transfer_predictions.csv.gz', index=False)
    write_json(out/'prediction_manifest.json', {'model_sha256': provenance, 'rows': len(predictions),
               'pilot': selection['pilot'], 'outcomes_used_for_fit_or_selection': False})
    finish_stage(out)


def target(data, transfer_dir, config, out):
    transfer_manifest = verify_stage(transfer_dir)
    require(transfer_manifest['contract']['data_sha256'] == sha256(data), 'Transfer data contract differs')
    transfer_info = json.loads((transfer_dir/'prediction_manifest.json').read_text())
    require(transfer_info['pilot'] == (config.get('pilot_households_per_fold') is not None), 'Pilot/full-run mismatch')
    contract = {'stage': 'target', 'data_sha256': sha256(data), 'config': config,
                'transfer_manifest_sha256': sha256(transfer_dir/'_SUCCESS.json')}
    if not start_stage(out, contract):
        return
    frames, spec = load_data(data)
    predictions = pd.read_csv(transfer_dir/'transfer_predictions.csv.gz')
    frame = frames['NLSY79'].merge(predictions[['pair_id', 'S_score', 'S_delta']], on='pair_id', how='left', validate='one_to_one')
    require(frame[['S_score', 'S_delta']].notna().all().all(), 'Missing transferred predictions')
    development = select_households(frame.loc[frame.analysis_role.eq('development')], config.get('pilot_households_per_fold'))
    test = frame.loc[frame.analysis_role.eq('locked_test')]
    sets = spec['features']['feature_sets']
    common = list(sets['common_primary29'])
    additional = list(sets['nlsy_additional82'])
    if config.get('exclude_education'):
        common = [c for c in common if not c.startswith('education4__')]
    variants = {'C': common, 'A': common+additional,
                'B_score': common+additional+['S_score'], 'B_change': common+additional+['S_delta']}
    output = test[['pair_id', 'person_id', 'household_id', 'followup_band', 'followup_years']].copy()
    summary = []
    for learner, grid in config['target_grids'].items():
        for variant, features in variants.items():
            model, score = train_one(development, features, learner, grid, config, False,
                                     out/learner/variant, learner+'_'+variant,
                                     {'transfer_manifest_sha256': sha256(transfer_dir/'_SUCCESS.json')})
            output[learner+'_'+variant] = model.predict(test.loc[:, features])
            summary.append({'learner': learner, 'variant': variant, **score})
    output.to_csv(out/'locked_predictions.csv.gz', index=False)
    write_json(out/'selection.json', {'development_cv': summary, 'primary_learner': config['primary_target_learner'],
               'primary_variant': config['primary_increment'], 'pilot': transfer_info['pilot'],
               'test_outcomes_used': False})
    finish_stage(out)


def evaluate(data, transfer_dir, target_dir, config, out):
    target_manifest = verify_stage(target_dir)
    verify_stage(transfer_dir)
    require(target_manifest['contract']['data_sha256'] == sha256(data), 'Evaluation data contract differs')
    require(target_manifest['contract']['transfer_manifest_sha256'] == sha256(transfer_dir/'_SUCCESS.json'), 'Wrong transfer run')
    require(target_manifest['contract']['config'] == config, 'Evaluation configuration differs from frozen training')
    require(config.get('pilot_households_per_fold') is None, 'Pilot runs cannot reveal locked-test performance')
    contract = {'stage': 'evaluation', 'data_sha256': sha256(data), 'config': config,
                'target_manifest_sha256': sha256(target_dir/'_SUCCESS.json')}
    if not start_stage(out, contract):
        return
    frames, _ = load_data(data)
    test = frames['NLSY79'].loc[lambda f: f.analysis_role.eq('locked_test')]
    downstream = pd.read_csv(target_dir/'locked_predictions.csv.gz')
    source_prediction = pd.read_csv(transfer_dir/'transfer_predictions.csv.gz')
    require(set(test.pair_id) == set(downstream.pair_id), 'Incomplete locked-test predictions')
    direct = ['S_score', 'trajectory_endpoint', 'simple_prediction', 'persistence_prediction']
    downstream_names = [c for c in downstream if c not in ['pair_id', 'person_id', 'household_id', 'followup_band', 'followup_years']]
    scored = test[['pair_id', 'person_id', 'household_id', 'followup_band', TARGET]].merge(
        downstream[['pair_id']+downstream_names], on='pair_id', validate='one_to_one').merge(
        source_prediction[['pair_id']+direct], on='pair_id', validate='one_to_one')
    rows = []
    for label in direct+downstream_names:
        for band in ['all']+sorted(scored.followup_band.unique()):
            subset = scored if band == 'all' else scored.loc[scored.followup_band.eq(band)]
            rows.append({'model': label, 'followup_band': band, **metrics(subset[TARGET], subset[label])})
    differences = []
    for learner in config['target_grids']:
        for variant in ['B_score', 'B_change']:
            differences.append({'learner': learner, 'comparison': 'A vs '+variant,
                                'primary': learner == config['primary_target_learner'] and variant == config['primary_increment'],
                                **paired_bootstrap(scored[TARGET], scored[learner+'_A'], scored[learner+'_'+variant],
                                                   scored.household_id, config['bootstrap_repeats'], config['seed'])})
    pd.DataFrame(rows).to_csv(out/'test_metrics.csv', index=False)
    pd.DataFrame(differences).to_csv(out/'paired_improvement.csv', index=False)
    scored.to_csv(out/'scored_test_predictions.csv.gz', index=False)
    write_json(out/'evaluation.json', {'locked_people': len(scored), 'locked_households': scored.household_id.nunique(),
               'primary_learner': config['primary_target_learner'], 'primary_increment': config['primary_increment'],
               'bootstrap': 'Paired household resampling of frozen test predictions', 'predictions_clipped': False})
    finish_stage(out)
