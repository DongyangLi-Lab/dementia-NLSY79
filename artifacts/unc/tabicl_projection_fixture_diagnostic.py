"""Capture private ICL projection fixtures during a coordinated exclusive GPU window.

This is a diagnostic, not a SHAP implementation. No model arithmetic is changed.
Each tensor is written separately; a complete 14,123-person capture is accepted
only when its original-input predictions pass the original numerical baseline.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

INPUT_SHA = '45d87bc03f17186a65feb4d0acfe79ebc768a1b882cae5689f20c3ea5d85680d'
SHORT_OUTPUT_SHA = 'a81e0af7f618308d7ef653513d89fd808280a103c765cb38a64499ad4d226eca'


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def save_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')
    tmp.replace(path)


def orchestrate():
    from artifacts.audit_tabicl_permutations import competitors
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise RuntimeError('Set CUDA_VISIBLE_DEVICES=0 during a coordinated exclusive window')
    if competitors():
        raise RuntimeError('Another TabICL process is running; pause the scheduler and waiting processes first')
    folder = ROOT / 'artifacts' / 'unc' / 'projection_fixtures_private'
    folder.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(folder).free < 16 * 1024**3:
        raise RuntimeError('At least 16 GiB free disk space is required')
    run = folder / datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run.mkdir(mode=0o700, exist_ok=False)
    manifest = {'status': 'running', 'script_sha256': file_hash(__file__), 'stages': {},
                'note': 'private participant-derived intermediate tensors; diagnostics only'}
    save_json(run / 'manifest.json', manifest)
    try:
        for stage in ('full', 'short', 'coalition'):
            if competitors(exclude_children=True):
                raise RuntimeError('Another TabICL process appeared before fixture capture')
            cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--stage', stage, '--run', str(run)]
            process = subprocess.Popen(cmd, cwd=ROOT)
            interference = []
            while process.poll() is None:
                interference.extend(competitors(exclude_children=True))
                time.sleep(1)
            if interference:
                save_json(run / (stage + '_interference.json'), interference)
                raise RuntimeError('Concurrent TabICL process detected during fixture capture')
            if process.returncode:
                raise RuntimeError(f'{stage} capture exited {process.returncode}; retained as unverified')
            result = json.loads((run / stage / 'result.json').read_text(encoding='utf-8'))
            manifest['stages'][stage] = result
            save_json(run / 'manifest.json', manifest)
        manifest['status'] = 'complete_diagnostic_only'
        save_json(run / 'manifest.json', manifest)
        print('FIXTURE_CAPTURE_COMPLETE', run, flush=True)
    except BaseException as e:
        manifest['status'] = 'failed_or_contended'
        manifest['error'] = str(e)
        save_json(run / 'manifest.json', manifest)
        raise


def capture(stage, run):
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from inference import evaluation, importance
    from inference.scoring import predict

    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise RuntimeError('Explicit exclusive GPU 0 selection is required')
    folder = run / stage
    folder.mkdir(mode=0o700, exist_ok=False)
    if stage == 'full':
        # Imports the existing fail-closed full-cohort batch/offload guards.
        from artifacts import tabicl_pinned_permutations as pinned
        model, _, frame, y, ids, _ = importance.load_temporal_model('TabICL', 6)
        saved = evaluation.load(6, 'TabICL', 'temporal')
        if len(frame) != 14123 or not np.array_equal(y, saved.y.to_numpy()) or not np.array_equal(ids.astype(str), saved.hhidpn.astype(str)):
            raise ValueError('Full-cohort identity or order differs from the saved baseline')
        query_rows = 14123
    elif stage == 'short':
        model, _, frame, _, _, _ = importance.load_temporal_model('TabICL', 6)
        source = ROOT / 'artifacts' / 'unc' / 'local_transform_diagnostic.npz'
        if file_hash(source) != INPUT_SHA:
            raise ValueError('128-query fixed transform hash does not match')
        z = np.load(source, allow_pickle=False)
        frame = frame.iloc[:128]
        for key, pp in model.ensemble_generator_.preprocessors_.items():
            arr = z[key][:128]
            pp.transform = lambda x, arr=arr: arr.copy() if len(x) == len(arr) else (_ for _ in ()).throw(ValueError('Unexpected fixed input shape'))
        query_rows = 128
    elif stage == 'coalition':
        from artifacts import tabicl_kernel_shap_sharded as shap
        model, dev, ext, columns, design = shap.load_inputs()
        bg = dev[design['background_indices']]
        xe = ext[design['explain_indices']]
        # Match the existing sequential SHAP control's warm-up order.
        p0 = predict(model, pd.DataFrame(bg, columns=columns))
        p1 = predict(model, pd.DataFrame(xe, columns=columns))
        mask = design['column_mask']
        array = np.where(mask[:, None, :], xe[0][None, None, :], bg[None, :, :]).reshape(-1, len(columns))
        if len(array) != 8000:
            raise ValueError('Coalition geometry differs from the approved design')
        np.save(folder / 'input.npy', array, allow_pickle=False)
        np.save(folder / 'warmup_bg.npy', p0, allow_pickle=False)
        np.save(folder / 'warmup_explain.npy', p1, allow_pickle=False)
        frame = pd.DataFrame(array, columns=columns)
        query_rows = 8000
    else:
        raise ValueError(stage)

    blocks = model.model_.icl_predictor.tf_icl.blocks
    projections = {id(block.attn.out_proj.weight): (i, block.attn.out_proj) for i, block in enumerate(blocks)}
    original_linear = F.linear
    records = []
    weights = {}
    count_by_block = {}
    index_path = folder / 'index.jsonl'
    index_file = index_path.open('x', encoding='utf-8')

    def save_tensor(path, tensor):
        # Only one CPU tensor copy is live at a time, including in full-cohort mode.
        data = tensor.detach().cpu().contiguous().numpy()
        raw_hash = hashlib.sha256(memoryview(data).cast('B')).hexdigest()
        with path.open('xb') as f:
            np.save(f, data, allow_pickle=False)
        del data
        return {'raw_sha256': raw_hash, 'file_sha256': file_hash(path), 'file': str(path.relative_to(run))}

    def meta(t):
        return {'shape': list(t.shape), 'stride': list(t.stride()), 'dtype': str(t.dtype),
                'device': str(t.device), 'storage_offset': int(t.storage_offset()),
                'contiguous': bool(t.is_contiguous()), 'data_ptr_alignment_16': int(t.data_ptr() % 16)}

    def intercepted(x, weight, bias=None):
        result = original_linear(x, weight, bias)
        target = projections.get(id(weight))
        if target is None or x.ndim != 3 or x.shape[-2] != query_rows:
            return result
        block, _ = target
        if shutil.disk_usage(folder).free < 2 * 1024**3:
            raise RuntimeError('Fixture capture stopped with less than 2 GiB free disk space')
        call = count_by_block.get(block, 0)
        count_by_block[block] = call + 1
        stem = f'block_{block:02d}_call_{call:03d}'
        if block not in weights:
            weights[block] = {'weight_meta': meta(weight), 'bias_meta': meta(bias),
                              'weight': save_tensor(folder / f'block_{block:02d}_weight.npy', weight),
                              'bias': save_tensor(folder / f'block_{block:02d}_bias.npy', bias)}
        record = {'stage': stage, 'block': block, 'call': call, 'input_meta': meta(x),
                  'output_meta': meta(result), 'autocast_enabled': torch.is_autocast_enabled('cuda'),
                  'autocast_dtype': str(torch.get_autocast_dtype('cuda')),
                  'input': save_tensor(folder / (stem + '_input.npy'), x),
                  'output': save_tensor(folder / (stem + '_output.npy'), result),
                  'weights': weights[block]}
        index_file.write(json.dumps(record, ensure_ascii=False) + '\n')
        index_file.flush()
        records.append({'block': block, 'call': call, 'input': record['input']['raw_sha256'],
                        'output': record['output']['raw_sha256']})
        print('FIXTURE', stage, block, call, record['input']['raw_sha256'], record['output']['raw_sha256'], flush=True)
        return result

    gpu_name = torch.cuda.get_device_name()
    env = {'gpu': gpu_name, 'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
           'torch': torch.__version__, 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
           'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
           'fp16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
           'fp16_accumulation': torch.backends.cuda.matmul.allow_fp16_accumulation,
           'preferred_blas_library': str(torch.backends.cuda.preferred_blas_library()),
           'memory_at_start': list(torch.cuda.mem_get_info()),
           'environment': {k: os.environ.get(k) for k in ('CUBLAS_WORKSPACE_CONFIG', 'PYTORCH_CUDA_ALLOC_CONF', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}}
    from artifact_store import job_dir
    model_path = job_dir('checkpoints', 6, 'final', 'TabICL') / 'temporal.pkl'
    env['model_sha256'] = file_hash(model_path)
    env['scoring_sha256'] = file_hash(ROOT / 'inference' / 'scoring.py')
    if stage == 'full':
        env['pinned_worker_sha256'] = file_hash(ROOT / 'artifacts' / 'tabicl_pinned_permutations.py')
    elif stage == 'short':
        env['transform_sha256'] = INPUT_SHA
    F.linear = intercepted
    try:
        p = predict(model, frame)
    finally:
        F.linear = original_linear
        index_file.close()
    if not records or set(count_by_block) != set(range(len(blocks))):
        raise ValueError(f'Incomplete projection capture: {count_by_block}')
    np.save(folder / 'predictions.npy', p, allow_pickle=False)
    result = {'status': 'diagnostic', 'stage': stage, 'query_rows': query_rows, 'records': len(records),
              'calls_per_block': count_by_block, 'index_sha256': file_hash(index_path),
              'predictions_sha256': file_hash(folder / 'predictions.npy'), 'environment': env}
    if stage == 'full':
        base = saved.p.to_numpy()
        error = np.abs(p - base)
        result['original_full_baseline'] = {'rtol': 1e-5, 'atol': 1e-7,
            'passed': bool(np.allclose(p, base, rtol=1e-5, atol=1e-7)),
            'max_abs': float(error.max()), 'mismatches': int((error > 1e-7 + 1e-5 * np.abs(base)).sum())}
        if not result['original_full_baseline']['passed']:
            save_json(folder / 'result.json', result)
            raise ValueError('Full original-input baseline failed; do not use captured fixtures as validated')
    elif stage == 'short':
        first = next((r for r in records if r['block'] == 0 and r['call'] == 0), None)
        result['independent_short_reference_matched'] = bool(first and first['output'] == SHORT_OUTPUT_SHA)
        if not result['independent_short_reference_matched']:
            save_json(folder / 'result.json', result)
            raise ValueError('128-query projection differs from the independent local reference')
    else:
        result['subject_index'] = 0
        result['design_sha256'] = file_hash(ROOT / 'artifacts' / 'tabicl_kernel_shap_design.npz')
        result['input_sha256'] = file_hash(folder / 'input.npy')
        ref = ROOT / 'artifacts' / 'tabicl_kernel_shap_sharded' / 'sequential_reference.npz'
        result['sequential_reference'] = 'not yet available'
        if shap.verified(ref.with_suffix('.done.json'), shap.stamp() | {'indices': [0, 1]}, (ref,)):
            with np.load(ref, allow_pickle=False) as zref:
                ref_values = zref['predictions'][0].reshape(-1)
                passed = (np.allclose(p0, zref['p0'], rtol=1e-5, atol=1e-7)
                          and np.allclose(p1, zref['p1'], rtol=1e-5, atol=1e-7)
                          and np.allclose(p, ref_values, rtol=1e-5, atol=1e-7))
                result['sequential_reference'] = {'passed': bool(passed), 'reference_sha256': file_hash(ref),
                                                   'max_abs': float(np.max(np.abs(p - ref_values)))}
                if not passed:
                    save_json(folder / 'result.json', result)
                    raise ValueError('8,000-row capture differs from the verified sequential local reference')
        result['note'] = 'Local control only; a missing sequential reference leaves the coalition comparison pending.'
    save_json(folder / 'result.json', result)
    print('FIXTURE_STAGE_COMPLETE', stage, json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('full', 'short', 'coalition'))
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    if args.stage:
        if args.run is None:
            parser.error('--stage requires --run')
        capture(args.stage, args.run)
    else:
        orchestrate()
