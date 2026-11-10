"""Replay TabICL checkpoints in fresh processes; preserve query and ensemble batches."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from artifact_store import job_dir, load_model, save_json, save_receipt, verified
from inference.scoring import metrics, predict
from train.cohorts import ROOT, ARTIFACTS, atomic_csv, load_cohorts, sha256
from train.imputation import cache_stamp, read_cache

SCRIPT = Path(__file__).resolve()


def selection_and_stamp(horizon, outcome):
    trained = job_dir('checkpoints', horizon, outcome, 'TabICL')
    done = trained / 'training.done.json'
    train_stamp = json.loads(done.read_text(encoding='utf-8'))['stamp']
    files = [trained / n for n in ('selection.json', 'validation_grid.csv', 'temporal.pkl')]
    if train_stamp['imputation'] != cache_stamp(horizon, outcome) or not verified(done, train_stamp, files):
        raise ValueError('Training receipt or imputation does not match')
    selection = json.loads(files[0].read_text(encoding='utf-8'))
    stamp = {'training': sha256(done), 'inference': sha256(ROOT / 'inference/predict.py'),
             'isolated_worker': sha256(SCRIPT)}
    for rep in selection['per_repeat']:
        key = f"rep{rep['repeat']}"
        receipt = trained / f'{key}.done.json'
        if not verified(receipt, train_stamp, (trained / f'{key}.json', trained / f'{key}.pkl')):
            raise ValueError(f'Invalid checkpoint receipt: {receipt}')
        stamp[key] = sha256(receipt)
    return trained, selection, stamp


def worker(horizon, outcome, stage):
    trained, selection, stamp = selection_and_stamp(horizon, outcome)
    dev, ext, predictors = load_cohorts(horizon, outcome)
    stage_key = 'external' if stage == 'temporal' else stage
    c = read_cache(horizon, outcome, stage_key, cache_stamp(horizon, outcome))
    if stage == 'temporal':
        model_file, X, cohort, selected, setting = trained / 'temporal.pkl', c['X_ext'], ext, selection['temporal'], 'temporal'
    else:
        r = int(stage.removeprefix('rep'))
        selected = next(item for item in selection['per_repeat'] if item['repeat'] == r)
        model_file, X, cohort, setting = trained / f'{stage}.pkl', c['X_test'], dev.iloc[c['test']], 'internal'
    model = load_model(model_file)
    start = time.perf_counter()
    p = predict(model, X)
    seconds = time.perf_counter() - start
    y = cohort.y.to_numpy()
    complete = cohort[predictors].notna().all(axis=1).to_numpy()
    repeat = selected.get('repeat', -1)
    rows = []
    for population, mask in (('all', np.ones(len(y), bool)), ('complete', complete)):
        rows.append({'setting': setting, 'population': population, 'repeat': repeat,
                     'config': selected['config'], **metrics(y[mask], p[mask], selected['threshold']),
                     'fit_seconds': selected['fit_seconds'], 'prediction_seconds': seconds,
                     'restarts': selected['restarts'], 'collapsed': selected['collapsed']})
    predictions = pd.DataFrame({'hhidpn': cohort.hhidpn.to_numpy(), 'y': y, 'p': p, 'complete': complete})
    if repeat >= 0:
        predictions.insert(0, 'repeat', repeat)
    target = ARTIFACTS / 'isolated_tabicl' / outcome / f'h{horizon}'
    target.mkdir(parents=True, exist_ok=True)
    paths = [target / f'{stage}-predictions.csv', target / f'{stage}-metrics.csv']
    atomic_csv(predictions, paths[0])
    atomic_csv(pd.DataFrame(rows), paths[1])
    save_receipt(target / f'{stage}.done.json', stamp, paths)
    print(f'{outcome}/h{horizon}/TabICL/{stage}: complete', flush=True)


def main(horizon, outcome, control):
    _, selection, stamp = selection_and_stamp(horizon, outcome)
    if control:
        subprocess.run([sys.executable, '-u', str(SCRIPT), '--worker', 'rep0', '--horizon', str(horizon), '--outcome', outcome], check=True)
        tmp = ARTIFACTS / 'isolated_tabicl' / outcome / f'h{horizon}' / 'rep0-predictions.csv'
        actual = pd.read_csv(tmp)
        saved = pd.read_csv(job_dir('predictions', horizon, outcome, 'TabICL') / 'test_predictions.csv')
        saved = saved[saved['repeat'] == 0].reset_index(drop=True)
        pd.testing.assert_frame_equal(actual[['repeat', 'hhidpn', 'y', 'complete']], saved[['repeat', 'hhidpn', 'y', 'complete']], check_dtype=False)
        delta = np.abs(actual.p.to_numpy() - saved.p.to_numpy())
        audit = {'rows': len(actual), 'max_absolute_difference': float(delta.max()),
                 'allclose_atol_1e-7_rtol_1e-5': bool(np.allclose(actual.p, saved.p, atol=1e-7, rtol=1e-5)),
                 'script_sha256': sha256(SCRIPT)}
        save_json(ARTIFACTS / 'isolated_tabicl' / 'control.json', audit)
        print(json.dumps(audit), flush=True)
        if not audit['allclose_atol_1e-7_rtol_1e-5']:
            raise ValueError('Isolated replay does not reproduce saved predictions')
        return
    audit = json.loads((ARTIFACTS / 'isolated_tabicl' / 'control.json').read_text(encoding='utf-8'))
    if not audit['allclose_atol_1e-7_rtol_1e-5'] or audit['script_sha256'] != sha256(SCRIPT):
        raise ValueError('An up-to-date isolated replay control is required')
    target = ARTIFACTS / 'isolated_tabicl' / outcome / f'h{horizon}'
    stages = [f"rep{x['repeat']}" for x in selection['per_repeat']] + ['temporal']
    for stage in stages:
        paths = [target / f'{stage}-predictions.csv', target / f'{stage}-metrics.csv']
        if not verified(target / f'{stage}.done.json', stamp, paths):
            subprocess.run([sys.executable, '-u', str(SCRIPT), '--worker', stage, '--horizon', str(horizon), '--outcome', outcome], check=True)
    predictions = job_dir('predictions', horizon, outcome, 'TabICL')
    predictions.mkdir(parents=True, exist_ok=True)
    files = [predictions / name for name in ('test_predictions.csv', 'temporal_predictions.csv', 'metrics.csv')]
    atomic_csv(pd.concat([pd.read_csv(target / f'{stage}-predictions.csv') for stage in stages[:-1]], ignore_index=True), files[0])
    atomic_csv(pd.read_csv(target / 'temporal-predictions.csv'), files[1])
    atomic_csv(pd.concat([pd.read_csv(target / f'{stage}-metrics.csv') for stage in stages], ignore_index=True), files[2])
    save_receipt(predictions / 'inference.done.json', stamp, files)
    print(f'{outcome}/h{horizon}/TabICL isolated held-out inference complete', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon', type=int, required=True)
    parser.add_argument('--outcome', choices=('final', 'any'), required=True)
    parser.add_argument('--worker')
    parser.add_argument('--control', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args.horizon, args.outcome, args.worker)
    else:
        main(args.horizon, args.outcome, args.control)
