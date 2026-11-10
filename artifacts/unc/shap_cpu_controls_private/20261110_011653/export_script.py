"""Export CPU-only, exact input transforms for cross-device SHAP controls.

These are diagnostic inputs, not SHAP results. No model prediction is called.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

from datetime import datetime
import hashlib
import io
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
from artifact_store import save_json
from artifacts import tabicl_kernel_shap_sharded as core
from train.cohorts import sha256


def array_info(value):
    value = np.ascontiguousarray(value)
    return {'shape': list(value.shape), 'dtype': str(value.dtype),
            'sha256': hashlib.sha256(value.tobytes()).hexdigest()}


def main():
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    stamp = core.stamp()
    # The original pickle includes CUDA storages. Remap them only in this
    # CPU-only diagnostic process; preserve all tensor values and the checkpoint.
    original_loader = torch.storage._load_from_bytes
    torch.storage._load_from_bytes = lambda data: torch.load(
        io.BytesIO(data), map_location='cpu', weights_only=False)
    try:
        model, dev, ext, columns, design = core.load_inputs()
    finally:
        torch.storage._load_from_bytes = original_loader
    bg = dev[design['background_indices']]
    xe = ext[design['explain_indices']]
    frames = {'full': ext, 'background': bg, 'explain': xe}
    for i in (0, 1):
        frames[f'subject_{i:03d}'] = np.where(
            design['column_mask'][:, None, :], xe[i][None, None, :], bg[None, :, :]
        ).reshape(-1, len(columns))
    folder = Path(__file__).resolve().parent / 'shap_cpu_controls_private' / datetime.now().strftime('%Y%m%d_%H%M%S')
    folder.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'scope': 'CPU preprocessing inputs only; no GPU or SHAP inference',
              'core_stamp': stamp, 'script_sha256': sha256(Path(__file__)),
              'columns': columns, 'cases': {}}
    generator = model.ensemble_generator_
    for name, values in frames.items():
        frame = pd.DataFrame(values, columns=columns)
        filtered = generator.unique_filter_.transform(model.X_encoder_.transform(frame))
        arrays = {key: pp.transform(filtered) for key, pp in generator.preprocessors_.items()}
        if name == 'full':
            previous = Path(__file__).resolve().parent / 'local_transform_diagnostic.npz'
            with np.load(previous) as prior:
                assert set(prior.files) == set(arrays)
                for key, value in arrays.items():
                    assert np.array_equal(prior[key], value, equal_nan=True), key
            report['full_matches_previous_export_sha256'] = sha256(previous)
        if name == 'subject_000':
            fixture = Path(__file__).resolve().parent / 'projection_fixtures_private' / '20261110_002014_320060_coalition_retry' / 'coalition' / 'result.json'
            import json
            expected = json.loads(fixture.read_text(encoding='utf-8'))['input_sha256']
            original_input = fixture.parent / 'input.npy'
            assert sha256(original_input) == expected, 'GPU fixture file hash changed'
            assert np.array_equal(values, np.load(original_input), equal_nan=True), 'Original GPU fixture input differs'
            report['subject_000_matches_gpu_fixture_sha256'] = sha256(fixture)
        file = folder / f'{name}.npz'
        np.savez_compressed(file, **arrays)
        with np.load(file) as saved:
            assert set(saved.files) == set(arrays)
            for key, value in arrays.items():
                assert np.array_equal(saved[key], value, equal_nan=True)
        report['cases'][name] = {'input': array_info(values), 'transforms': {
            key: array_info(value) for key, value in arrays.items()},
            'file': file.name, 'file_sha256': sha256(file)}
        print(f'{name}: {len(values)} rows, {file.stat().st_size} bytes', flush=True)
    # Training transforms are frozen in the pickle; export hashes for verification.
    report['training_transforms'] = {key: array_info(pp.X_transformed_)
                                     for key, pp in generator.preprocessors_.items()}
    assert not torch.cuda.is_initialized(), 'CPU exporter unexpectedly initialized CUDA'
    assert core.stamp() == stamp, 'Sources changed during export'
    report['status'] = 'exported_and_readback_verified'
    report['cuda_initialized'] = torch.cuda.is_initialized()
    save_json(folder / 'manifest.json', report)
    print(folder / 'manifest.json', flush=True)


if __name__ == '__main__':
    main()
