"""Run the unchanged evaluation functions by outcome/horizon/population in parallel."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import pickle
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd
from artifact_store import save_json, write_file
from model.specs import MODELS
from train.cohorts import ARTIFACTS, sha256
from inference import evaluation as ev

SCRIPT = Path(__file__).resolve()
ROOT = ARTIFACTS / 'parallel_evaluation'
NAMES = ('internal_metrics', 'internal_pairwise', 'temporal_metrics', 'temporal_pairwise',
         'calibration_curves', 'decision_curves')


def input_stamp(outcome):
    items = {}
    for h in (6, 8):
        for model in MODELS:
            for prefix, name in (('predictions', 'inference.done.json'), ('checkpoints', 'selection.json')):
                path = ARTIFACTS / prefix / outcome / f'h{h}' / model / name
                items[str(path.relative_to(ARTIFACTS))] = sha256(path)
    return {'outcome': outcome, 'evaluation_sha256': sha256(Path(ev.__file__)),
            'worker_sha256': sha256(SCRIPT), 'inputs': items, 'bootstrap_replicates': ev.N_BOOT}


def save_fragment(path, obj):
    def write(tmp):
        with tmp.open('wb') as stream:
            pickle.dump(obj, stream, protocol=pickle.HIGHEST_PROTOCOL)
    write_file(path, write)


def run_cell(outcome, cell):
    ev.OUTCOME = outcome
    started = time.time()
    if cell == 'horizon':
        data = {'horizon_comparison': ev.horizon_table()}
    else:
        h_text, population = cell.split('_')
        h = int(h_text[1:])
        a, b = ev.internal_table(h, population)
        c, d = ev.temporal_table(h, population)
        data = dict(zip(NAMES, (a, b, c, d, ev.calibration_curves(h, population), ev.decision_curves(h, population))))
    path = ROOT / outcome / f'{cell}.pkl'
    path.parent.mkdir(parents=True, exist_ok=True)
    save_fragment(path, {'stamp': input_stamp(outcome), 'data': data})
    print(f'{outcome}/{cell} complete in {(time.time()-started)/60:.1f} min', flush=True)


def run_all(workers):
    ROOT.mkdir(parents=True, exist_ok=True)
    tasks = [(o, c) for o in ('final', 'any') for c in ('h6_all', 'h6_complete', 'h8_all', 'h8_complete', 'horizon')]
    stamps = {o: input_stamp(o) for o in ('final', 'any')}
    def execute(task):
        outcome, cell = task
        p = ROOT / outcome / f'{cell}.pkl'
        if p.exists():
            with p.open('rb') as stream:
                if pickle.load(stream)['stamp'] == stamps[outcome]:
                    return f'{outcome}/{cell} reused'
        log_path = ROOT / f'{outcome}-{cell}.log'
        with log_path.open('a', encoding='utf-8') as log:
            run = subprocess.run([sys.executable, '-u', str(SCRIPT), '--outcome', outcome, '--cell', cell],
                                 cwd=ev.ARTIFACTS.parent, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
        if run.returncode:
            raise RuntimeError(f'{outcome}/{cell} failed (exit {run.returncode}): {log_path}')
        return f'{outcome}/{cell} complete'
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(execute, task) for task in tasks]
        for future in as_completed(futures):
            print(future.result(), flush=True)
    for outcome in ('final', 'any'):
        ev.OUTCOME = outcome
        prefix = '' if outcome == 'final' else 'sensitivity_any_'
        cells = []
        for cell in ('h6_all', 'h6_complete', 'h8_all', 'h8_complete'):
            with (ROOT / outcome / f'{cell}.pkl').open('rb') as stream:
                obj = pickle.load(stream)
            if obj['stamp'] != input_stamp(outcome):
                raise ValueError(f'Stale {outcome}/{cell} fragment')
            cells.append(obj['data'])
        with (ROOT / outcome / 'horizon.pkl').open('rb') as stream:
            horizon = pickle.load(stream)
        if horizon['stamp'] != input_stamp(outcome):
            raise ValueError(f'Stale {outcome}/horizon fragment')
        for name in NAMES:
            pd.concat([cell[name] for cell in cells], ignore_index=True).to_csv(ev.TABLES / f'{prefix}{name}.csv', index=False)
        pd.concat([ev.runtime_table(h) for h in (6, 8)], ignore_index=True).to_csv(ev.TABLES / f'{prefix}runtime.csv', index=False)
        horizon['data']['horizon_comparison'].to_csv(ev.TABLES / f'{prefix}horizon_comparison.csv', index=False)
        save_json(ROOT / f'{outcome}-complete.json', {'stamp': input_stamp(outcome), 'cells': 5})
        print(f'{outcome} evaluation tables merged', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outcome', choices=('final', 'any'))
    parser.add_argument('--cell')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if args.cell:
        run_cell(args.outcome, args.cell)
    else:
        run_all(args.workers)
