"""A small standalone linear-operator comparison on GPU1; no classifier is loaded."""
import hashlib
import json
import os
from pathlib import Path
import time

if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
    raise RuntimeError('This diagnostic must use physical GPU1 exclusively')
import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
if torch.cuda.device_count() != 1 or '3070 Ti' not in torch.cuda.get_device_name(0):
    raise RuntimeError('Visible device is not the intended RTX 3070 Ti')
root = Path(__file__).resolve().parent
fixture = root / 'projection_fixtures_private' / '20261110_002014_320060'
record = json.loads((fixture / 'short' / 'index.jsonl').read_text().splitlines()[0])
def load(entry):
    path = fixture / entry['file']
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry['file_sha256']:
        raise ValueError(f'Fixture hash mismatch: {path}')
    return np.load(path, allow_pickle=False)
x = torch.from_numpy(load(record['input'])).to('cuda:0')
w = torch.from_numpy(load(record['weights']['weight'])).to('cuda:0')
b = torch.from_numpy(load(record['weights']['bias'])).to('cuda:0')
expected = load(record['output'])
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_fp16_accumulation = False
cases = {}
with torch.no_grad():
    for library in ('cublas', 'cublaslt'):
        torch.backends.cuda.preferred_blas_library(library)
        for reduced in (True, False):
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = reduced
            for rows in (None, 128, 64, 32):
                name = f'{library}_reduced{reduced}_rows{rows}'
                started = time.monotonic()
                with torch.autocast('cuda', dtype=torch.float16):
                    if rows is None:
                        result = F.linear(x, w, b)
                    else:
                        flat = x.reshape(-1, x.shape[-1])
                        result = torch.cat([F.linear(z, w, b) for z in flat.split(rows)]).reshape_as(x)
                actual = result.cpu().numpy()
                diff = np.abs(actual.astype('f') - expected.astype('f'))
                cases[name] = {'sha256': hashlib.sha256(actual.tobytes()).hexdigest(),
                               'exact': bool(np.array_equal(actual, expected)),
                               'max_abs': float(diff.max()),
                               'mismatches': int(np.count_nonzero(diff > 1e-7 + 1e-5 * np.abs(expected.astype('f')))),
                               'seconds': time.monotonic() - started}
                print(name, json.dumps(cases[name]), flush=True)
report = {'scope': 'Latent linear operator, not dementia prediction probabilities or end-to-end validation',
          'device': torch.cuda.get_device_name(0), 'visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
          'torch': torch.__version__, 'cuda': torch.version.cuda,
          'reference_sha256': record['output']['raw_sha256'],
          'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'cases': cases}
(root / 'projection_device1_comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
