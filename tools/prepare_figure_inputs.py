"""Export lightweight saved fit diagnostics required for result-only plotting.

This is a packaging step, not a fitting/prediction step. No pickle is opened.
"""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def prepare():
    dest = ROOT / 'artifacts/figure_inputs'
    records = []
    for source in sorted((ROOT / 'artifacts/checkpoints/final').glob('h*/*/*')):
        if source.name not in ('validation_grid.csv', 'selection.json'):
            continue
        target = dest / 'selection/final' / source.relative_to(ROOT / 'artifacts/checkpoints/final')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records.append({'path': target.relative_to(dest).as_posix(),
                        'source': source.relative_to(ROOT).as_posix(),
                        'sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
    assert len(records) == 32, f'Expected 8 models x 2 horizons x 2 files, found {len(records)}'
    (dest / 'manifest.json').write_text(json.dumps({'files': records,
        'scope': 'Saved final-outcome validation timing and threshold selections; no fitted models.'}, indent=2), encoding='utf-8')
    print(f'Prepared {len(records)} lightweight plotting inputs')


if __name__ == '__main__':
    prepare()
