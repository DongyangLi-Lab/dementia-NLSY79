"""Command-line entry point: python -m part2_models --help."""
import argparse
import json
from pathlib import Path

from . import workflow
from .data import ROOT, load_data, require, sha256
from .store import load_model, verify_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['validate', 'source', 'transfer', 'target', 'evaluate', 'predict'])
    parser.add_argument('--data', type=Path, required=True, help='Frozen Stage 02 ZIP')
    parser.add_argument('--config', type=Path, default=ROOT/'planning/experiment_config.json')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--source', type=Path, help='Completed source stage directory')
    parser.add_argument('--transfer', type=Path, help='Completed transfer stage directory')
    parser.add_argument('--target', type=Path, help='Completed downstream stage directory')
    parser.add_argument('--model', type=Path, help='Saved model directory for prediction replay')
    parser.add_argument('--cohort', choices=['HRS', 'NLSY79'], default='NLSY79')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.action == 'validate':
        frames, _ = load_data(args.data)
        print(json.dumps({name: {'rows': len(f), 'people': f.person_id.nunique(), 'columns': len(f.columns)}
                          for name, f in frames.items()}, indent=2))
        return
    if args.out is None:
        parser.error('--out is required')
    required = {'transfer': ['source'], 'target': ['transfer'], 'evaluate': ['transfer', 'target'], 'predict': ['model']}
    for name in required.get(args.action, []):
        if getattr(args, name) is None:
            parser.error('--'+name+' is required')
    if args.action == 'source':
        workflow.source(args.data, config, args.out)
    elif args.action == 'transfer':
        workflow.transfer(args.data, args.source, args.out)
    elif args.action == 'target':
        workflow.target(args.data, args.transfer, config, args.out)
    elif args.action == 'evaluate':
        workflow.evaluate(args.data, args.transfer, args.target, config, args.out)
    elif args.action == 'predict':
        frames, _ = load_data(args.data)
        frame = frames[args.cohort]
        model, info = load_model(args.model)
        if 'S_score' in model.features or 'S_delta' in model.features:
            import pandas as pd
            if args.transfer is None or args.cohort != 'NLSY79':
                parser.error('Downstream B prediction requires --cohort NLSY79 and --transfer')
            manifest = verify_stage(args.transfer)
            require(manifest['contract']['data_sha256'] == sha256(args.data), 'Wrong transfer data')
            require(info['transfer_manifest_sha256'] == sha256(args.transfer/'_SUCCESS.json'), 'Wrong transferred-feature model version')
            transferred = pd.read_csv(args.transfer/'transfer_predictions.csv.gz')
            frame = frame.merge(transferred[['pair_id', 'S_score', 'S_delta']], on='pair_id', how='left', validate='one_to_one')
            require(frame[['S_score', 'S_delta']].notna().all().all(), 'Missing transferred predictions')
        if args.out.exists():
            raise FileExistsError(args.out)
        result = frame[['pair_id', 'person_id']].copy()
        result['prediction'] = model.predict(frame)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.out, index=False)


if __name__ == '__main__':
    main()
