"""Validate and import one UNC comparison shard without replacing local results."""
import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from artifact_store import job_dir, write_file
from train.cohorts import load_cohorts, sha256
from train.imputation import cache_stamp, read_cache


def receive(archive):
    with zipfile.ZipFile(archive) as z:
        meta = json.loads(z.read('manifest.json'))
        model, repeat = meta['model'], meta['repeat']
        if model not in ('GAT', 'TabICL') or repeat not in range(10) or (model == 'GAT' and repeat == 0):
            raise ValueError('Shard outside assignment')
        names = [f'{s}-rep{repeat}{suffix}' for s in ('mean', 'complete_case')
                 for suffix in ('.csv', '-predictions.csv', '.done.json')]
        if set(z.namelist()) != set(names + ['manifest.json']) or set(meta['files']) != set(names):
            raise ValueError('Unexpected shard contents')
        blobs = {n: z.read(n) for n in names}
        if any(hashlib.sha256(blobs[n]).hexdigest() != meta['files'][n] for n in names):
            raise ValueError('Shard checksum mismatch')
        dev, _, inputs = load_cohorts(6)
        c = read_cache(6, 'final', f'rep{repeat}', cache_stamp(6, 'final'))
        for strategy in ('mean', 'complete_case'):
            result = f'{strategy}-rep{repeat}.csv'
            raw = f'{strategy}-rep{repeat}-predictions.csv'
            done = f'{strategy}-rep{repeat}.done.json'
            receipt = json.loads(blobs[done])
            if receipt['stamp']['implementation'] != sha256(Path('train/imputation_comparison.py')):
                raise ValueError('Comparison implementation differs')
            if receipt['sha256'] != {result: meta['files'][result], raw: meta['files'][raw]}:
                raise ValueError('Receipt mismatch')
            actual = pd.read_csv(io.BytesIO(blobs[raw]), dtype={'hhidpn': str})
            ix = c['test']
            if strategy == 'complete_case':
                ix = ix[dev[inputs].notna().all(axis=1).to_numpy()[ix]]
            expected = dev.iloc[ix]
            if not np.array_equal(actual.hhidpn.astype(str).to_numpy(), expected.hhidpn.astype(str).to_numpy()):
                raise ValueError('Participant identifiers differ from original test split')
            if not np.array_equal(actual.y.to_numpy(), expected.y.to_numpy()):
                raise ValueError('Outcomes differ from original test split')
            if not np.isfinite(actual.p).all() or not actual.p.between(0, 1).all():
                raise ValueError('Invalid probabilities')
            row = pd.read_csv(io.BytesIO(blobs[result]))
            if len(row) != 1 or row.iloc[0]['model'] != model or row.iloc[0]['strategy'] != strategy or int(row.iloc[0]['repeat']) != repeat or int(row.iloc[0]['n']) != len(ix):
                raise ValueError('Result metadata differs')
        target = job_dir('imputation_comparison', 6, 'final', model)
        for n in names:
            path = target / n
            if path.exists() and sha256(path) != meta['files'][n]:
                raise FileExistsError(f'Conflicting existing result: {path}')
        for n in sorted(names, key=lambda n: n.endswith('.done.json')):
            path = target / n
            if not path.exists():
                write_file(path, lambda p, data=blobs[n]: p.write_bytes(data))
    print(f'Imported {model} rep{repeat} (mean and complete case), UNC job {meta["job"]}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    receive(parser.parse_args().archive)
