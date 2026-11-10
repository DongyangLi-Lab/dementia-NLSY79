"""Run unmodified imputation comparisons for explicitly assigned repeats only."""
import argparse
import os
import subprocess
import sys
from pathlib import Path
from artifact_store import save_json
from train.cohorts import ARTIFACTS, sha256

SCRIPT = Path(__file__).resolve()


def worker(model, device, repeat):
    import train.imputation_comparison as comparison
    original_range = range
    def restricted(n):
        if n != comparison.N_REPEATS:
            return original_range(n)
        return [repeat]
    comparison.range = restricted
    print(f'{model} repeat {repeat}: running both strategies', flush=True)
    comparison.run(model, device)


def main(model, device, repeats):
    audit = ARTIFACTS / 'local_comparison_splits' / f'{model}-gpu{device}.json'
    save_json(audit, {'model': model, 'device': device, 'assigned_repeats': repeats,
                      'source_sha256': sha256(SCRIPT),
                      'implementation_sha256': sha256(ARTIFACTS.parent / 'train/imputation_comparison.py'),
                      'status': 'running'})
    for r in repeats:
        subprocess.run([sys.executable, '-u', str(SCRIPT), '--model', model, '--device', device, '--worker', str(r)],
                       cwd=ARTIFACTS.parent, env=os.environ.copy(), check=True)
    save_json(audit, {'model': model, 'device': device, 'assigned_repeats': repeats,
                      'source_sha256': sha256(SCRIPT),
                      'implementation_sha256': sha256(ARTIFACTS.parent / 'train/imputation_comparison.py'),
                      'status': 'complete'})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, choices=('GAT', 'TabICL'))
    p.add_argument('--device', required=True)
    p.add_argument('--repeats', type=int, nargs='+')
    p.add_argument('--worker', type=int)
    a = p.parse_args()
    if a.worker is not None:
        worker(a.model, a.device, a.worker)
    elif a.repeats and all(0 <= r < 10 for r in a.repeats) and len(set(a.repeats)) == len(a.repeats):
        main(a.model, a.device, a.repeats)
    else:
        raise ValueError('Specify unique repeat indices from 0 to 9')
