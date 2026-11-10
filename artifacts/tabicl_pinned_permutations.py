"""Pin the observed, baseline-matching full-cohort execution geometry.

Only supports the 14,123-person temporal permutation shape. Fails on an
unexpected geometry or OOM fallback instead of silently changing arithmetic.
"""
import argparse
import importlib.util
import os
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tabicl.model.inference import InferenceManager

original_estimate = InferenceManager.estimate_safe_batch_size
original_call = InferenceManager.__call__
original_split = InferenceManager.compute_split_sizes

def estimate(self, seq_len, *args, **kwargs):
    available, _ = original_estimate(self, seq_len, *args, **kwargs)
    expected = {'tf_col': (27369, 88), 'tf_row': (98, 16456), 'tf_icl': (27369, 17)}
    length, batch = expected[self.enc_name]
    if seq_len != length:
        raise ValueError(f'Unexpected pinned geometry {self.enc_name}: {seq_len}')
    return available, batch

def call(self, *args, **kwargs):
    self.offload = self.enc_name == 'tf_col'
    return original_call(self, *args, **kwargs)

def split(dims, batch):
    expected = {(98,): 88, (8, 27369): 16456}
    if tuple(dims) not in expected or batch != expected[tuple(dims)]:
        raise ValueError(f'Pinned execution changed or attempted OOM fallback: {dims}, {batch}')
    return original_split(dims, batch)

InferenceManager.estimate_safe_batch_size = estimate
InferenceManager.__call__ = call
InferenceManager.compute_split_sizes = staticmethod(split)

spec = importlib.util.spec_from_file_location('permutations', Path(__file__).with_name('tabicl_local_permutation_shard.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.SCRIPT = Path(__file__).resolve()

# Preserve invalid output evidence before the shared scorer rejects it. The
# returned object and arithmetic are unchanged; failed predictions stay invalid.
original_inputs = module.inputs

def inputs_with_diagnostics():
    model, frame, groups = original_inputs()
    predict_proba = model.predict_proba
    def predict_with_evidence(x):
        result = predict_proba(x)
        p = module.np.asarray(result)[:, 1]
        if len(p) != len(x) or not module.np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            folder = module.WORK / 'invalid_predictions' / f'{time.time_ns()}_{os.getpid()}'
            folder.mkdir(parents=True, exist_ok=False)
            module.np.savez_compressed(folder / 'evidence.npz',
                                      probabilities=result, inputs=x.to_numpy(), columns=x.columns.to_numpy())
            module.save_json(folder / 'evidence.json', {
                'argv': sys.argv, 'worker_sha256': module.sha256(Path(__file__)),
                'rows': len(x), 'nan': int(module.np.isnan(p).sum()),
                'inf': int(module.np.isinf(p).sum()),
                'out_of_range': int(((p < 0) | (p > 1)).sum()),
                'artifact_sha256': module.sha256(folder / 'evidence.npz')})
            print(f'Invalid prediction evidence retained: {folder}', flush=True)
        return result
    model.predict_proba = predict_with_evidence
    return model, frame, groups

module.inputs = inputs_with_diagnostics

if __name__ == '__main__':
    from artifacts.audit_tabicl_permutations import competitors
    live = competitors()
    if live:
        raise RuntimeError(f'Pinned execution requires an exclusive GPU window: {live}')
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int)
    parser.add_argument('--replicate', type=int, default=0)
    parser.add_argument('--indices', type=int, nargs='+')
    args = parser.parse_args()
    if args.worker is not None:
        module.worker(args.worker, args.replicate)
    elif args.indices and len(set(args.indices)) == len(args.indices) and all(0 <= i < 60 for i in args.indices):
        module.main(args.indices)
    else:
        parser.error('Specify worker or unique group indices')
