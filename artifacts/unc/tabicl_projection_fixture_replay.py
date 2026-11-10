"""Replay one captured ICL linear operator independently; diagnostic use only."""
import argparse
import hashlib
import json
from pathlib import Path, PureWindowsPath


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True, help='private fixture run directory')
    parser.add_argument('--stage', choices=('full', 'short', 'coalition'), required=True)
    parser.add_argument('--record', type=int, default=0, help='zero-based line in stage/index.jsonl')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--blas', choices=('recorded', 'cublas', 'cublaslt'), default='recorded')
    args = parser.parse_args()
    import numpy as np
    import torch
    import torch.nn.functional as F

    stage = args.run / args.stage
    result = json.loads((stage / 'result.json').read_text(encoding='utf-8'))
    if args.stage == 'full' and not result['original_full_baseline']['passed']:
        raise ValueError('Full-cohort source capture failed the original numerical baseline')
    if args.stage == 'short' and not result['independent_short_reference_matched']:
        raise ValueError('Short source capture failed its independent local reference')
    with (stage / 'index.jsonl').open(encoding='utf-8') as f:
        records = [json.loads(line) for line in f]
    if not 0 <= args.record < len(records):
        raise ValueError('Requested record index is outside the fixture')
    rec = records[args.record]
    if not rec['input_meta']['contiguous'] or not rec['output_meta']['contiguous']:
        raise ValueError('Noncontiguous source layout needs a stride-preserving replay')

    def read(entry):
        relative = PureWindowsPath(entry['file']) if '\\' in entry['file'] else Path(entry['file'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Fixture path must stay within the private run directory')
        path = args.run.joinpath(*relative.parts)
        if file_hash(path) != entry['file_sha256']:
            raise ValueError(f'Fixture hash mismatch: {path.name}')
        return np.load(path, mmap_mode='r', allow_pickle=False)

    source = read(rec['input'])
    weight = read(rec['weights']['weight'])
    bias = read(rec['weights']['bias'])
    expected = read(rec['output'])
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA device is required to replay the captured GPU operator')
    torch.backends.cuda.matmul.allow_tf32 = result['environment']['matmul_tf32']
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = result['environment']['fp16_reduced_precision_reduction']
    torch.backends.cuda.matmul.allow_fp16_accumulation = result['environment']['fp16_accumulation']
    recorded_blas = result['environment']['preferred_blas_library'].split('.')[-1].lower()
    selected_blas = recorded_blas if args.blas == 'recorded' else args.blas
    if selected_blas not in ('cublas', 'cublaslt', 'default'):
        raise ValueError(f'Unknown recorded BLAS library {recorded_blas}')
    torch.backends.cuda.preferred_blas_library(selected_blas)
    dtype_name = rec['autocast_dtype'].split('.')[-1]
    if dtype_name not in ('float16', 'bfloat16'):
        raise ValueError(f'Unexpected recorded autocast dtype {dtype_name}')
    x = torch.from_numpy(np.array(source, copy=True)).cuda()
    w = torch.from_numpy(np.array(weight, copy=True)).cuda()
    b = torch.from_numpy(np.array(bias, copy=True)).cuda()

    def call():
        with torch.autocast('cuda', enabled=rec['autocast_enabled'], dtype=getattr(torch, dtype_name)):
            return F.linear(x, w, b)
    if args.profile:
        import torch.profiler as profiler
        with profiler.profile(activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA]) as trace:
            actual = call()
        for event in trace.events():
            if 'CUDA' in str(event.device_type).upper() or any(s in event.name.lower() for s in ('addmm', 'gemm', 'matmul')):
                print('REPLAY_EVENT', str(event.device_type), event.name, flush=True)
    else:
        actual = call()
    actual = actual.detach().cpu().contiguous().numpy()
    if tuple(actual.shape) != tuple(expected.shape):
        raise ValueError('Replayed operator produced an unexpected shape')
    reference = np.asarray(expected).reshape(-1)
    observed = actual.reshape(-1)
    max_abs = 0.0
    unequal = 0
    mismatches = 0
    for start in range(0, observed.size, 262144):
        left = observed[start:start + 262144].astype(np.float32)
        right = reference[start:start + 262144].astype(np.float32)
        delta = np.abs(left - right)
        max_abs = max(max_abs, float(delta.max(initial=0)))
        unequal += int(np.count_nonzero(left != right))
        mismatches += int(np.count_nonzero(delta > 1e-7 + 1e-5 * np.abs(right)))
    print('REPLAY_RESULT', json.dumps({'stage': args.stage, 'record': args.record, 'block': rec['block'],
          'call': rec['call'], 'gpu': torch.cuda.get_device_name(), 'source_output_sha256': rec['output']['raw_sha256'],
          'blas': selected_blas,
          'observed_sha256': hashlib.sha256(memoryview(actual).cast('B')).hexdigest(), 'max_abs': max_abs,
          'unequal_elements': unequal, 'outside_rtol_1e_5_atol_1e_7': mismatches,
          'note': 'Operator diagnostic only; original full-prediction and coalition gates remain mandatory.'}), flush=True)


if __name__ == '__main__':
    main()
