"""Verify all 352 frozen fits and their prediction receipt links, without loading models."""
from datetime import datetime
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from artifact_store import job_dir, save_json, verified
from model.specs import MODELS
from train.cohorts import ARTIFACTS, sha256

def main():
    runs = []
    for outcome in ('final', 'any'):
        for horizon in (6, 8):
            for model in MODELS:
                folder = job_dir('checkpoints', horizon, outcome, model)
                train_path = folder / 'training.done.json'
                training = json.loads(train_path.read_text(encoding='utf-8'))
                files = [folder / name for name in ('selection.json', 'validation_grid.csv', 'temporal.pkl')]
                if not verified(train_path, training['stamp'], files):
                    raise ValueError(f'Training receipt mismatch: {train_path}')
                receipts = {'training.done.json': sha256(train_path)}
                models = {'temporal.pkl': sha256(folder / 'temporal.pkl')}
                for r in range(10):
                    receipt = folder / f'rep{r}.done.json'
                    files = [folder / f'rep{r}.json', folder / f'rep{r}.pkl']
                    if not verified(receipt, training['stamp'], files):
                        raise ValueError(f'Repeat receipt mismatch: {receipt}')
                    receipts[receipt.name] = sha256(receipt)
                    models[files[1].name] = json.loads(receipt.read_text())['sha256'][files[1].name]
                pred = job_dir('predictions', horizon, outcome, model)
                inference = json.loads((pred / 'inference.done.json').read_text())
                if inference['stamp']['training'] != receipts['training.done.json']:
                    raise ValueError(f'Prediction-to-training link mismatch: {pred}')
                files = [pred / name for name in ('test_predictions.csv', 'temporal_predictions.csv', 'metrics.csv')]
                if not verified(pred / 'inference.done.json', inference['stamp'], files):
                    raise ValueError(f'Prediction receipt mismatch: {pred}')
                runs.append({'outcome': outcome, 'horizon': horizon, 'model': model,
                             'checkpoint_sha256': models, 'receipt_sha256': receipts,
                             'inference_receipt_sha256': sha256(pred / 'inference.done.json')})
                print(f'{outcome}/h{horizon}/{model}: 11 fit artifacts and held-out predictions verified', flush=True)
    report = {'status': 'validated', 'observed_at': datetime.now().astimezone().isoformat(),
              'scope': 'Frozen checkpoint byte integrity, receipt consistency, and prediction links. Not a GPU replay or importance completion claim.',
              'implementation_sha256': sha256(Path(__file__)), 'run_count': len(runs),
              'fit_count': sum(len(x['checkpoint_sha256']) for x in runs), 'runs': runs}
    save_json(ARTIFACTS / 'frozen_checkpoint_integrity.json', report)
    print(f"Validated {report['fit_count']} frozen fits across {len(runs)} runs", flush=True)

if __name__ == '__main__':
    main()
