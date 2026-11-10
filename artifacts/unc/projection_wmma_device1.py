"""Run the small WMMA diagnostic on GPU1, or GPU0 during an exclusive window."""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import time
import sys
physical_device = os.environ.get('CUDA_VISIBLE_DEVICES')
if physical_device not in ('0', '1'):
    raise RuntimeError('Select exactly one physical GPU explicitly')
if physical_device == '0':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from artifacts.audit_tabicl_permutations import competitors
    if competitors():
        raise RuntimeError('GPU0 comparison requires an exclusive window')
import numpy as np
import torch
torch.set_num_threads(1)
expected_device = '5070 Ti' if physical_device == '0' else '3070 Ti'
if torch.cuda.device_count() != 1 or expected_device not in torch.cuda.get_device_name():
    raise RuntimeError('Unexpected GPU')
root = Path(__file__).resolve().parent
lib = Path(torch.__file__).parent / 'lib'
nvrtc = C.WinDLL(str(lib / 'nvrtc64_120_0.dll'))
driver = C.WinDLL('nvcuda.dll')
V, I, U, S = C.c_void_p, C.c_int, C.c_uint, C.c_size_t
def api(dll, name, args):
    f = getattr(dll, name); f.argtypes = args; f.restype = I
    return f
create = api(nvrtc, 'nvrtcCreateProgram', [C.POINTER(V), C.c_char_p, C.c_char_p, I, V, V])
compile_program = api(nvrtc, 'nvrtcCompileProgram', [V, I, C.POINTER(C.c_char_p)])
get_log_size = api(nvrtc, 'nvrtcGetProgramLogSize', [V, C.POINTER(S)])
get_log = api(nvrtc, 'nvrtcGetProgramLog', [V, V])
get_ptx_size = api(nvrtc, 'nvrtcGetPTXSize', [V, C.POINTER(S)])
get_ptx = api(nvrtc, 'nvrtcGetPTX', [V, V])
destroy = api(nvrtc, 'nvrtcDestroyProgram', [C.POINTER(V)])
load_module = api(driver, 'cuModuleLoadDataEx', [C.POINTER(V), V, U, V, V])
get_function = api(driver, 'cuModuleGetFunction', [C.POINTER(V), V, C.c_char_p])
launch = api(driver, 'cuLaunchKernel', [V, U, U, U, U, U, U, U, V, C.POINTER(V), V])
unload = api(driver, 'cuModuleUnload', [V])
def check(status, where):
    if status: raise RuntimeError(f'{where} failed: {status}')
source_path = root / 'projection_round128.cu'
source = source_path.read_bytes()
program = V()
check(create(C.byref(program), source, b'projection_round128.cu', 0, None, None), 'create')
capability = torch.cuda.get_device_capability()
architecture = f'compute_{capability[0]}{capability[1]}'
options = [f'--gpu-architecture={architecture}'.encode(), b'--std=c++17',
           b'--include-path=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/include']
result = compile_program(program, len(options), (C.c_char_p * len(options))(*options))
size = S(); check(get_log_size(program, C.byref(size)), 'log size')
log = C.create_string_buffer(size.value); check(get_log(program, log), 'log')
print(log.value.decode(errors='replace'), flush=True)
check(result, 'NVRTC compile')
check(get_ptx_size(program, C.byref(size)), 'PTX size')
ptx = C.create_string_buffer(size.value); check(get_ptx(program, ptx), 'PTX')
check(destroy(C.byref(program)), 'destroy program')
(root / f'projection_round128_{architecture}.ptx').write_bytes(ptx.raw)
torch.cuda.init()
context_anchor = torch.empty(1, device='cuda:0')  # materialize the CUDA primary context
module = V(); function = V()
check(load_module(C.byref(module), ptx, 0, None, None), 'load module')
check(get_function(C.byref(function), module, b'projection_periodic_round'), 'get function')
fixture = root / 'projection_fixtures_private' / '20261110_002014_320060'
record = json.loads((fixture / 'short' / 'index.jsonl').read_text().splitlines()[0])
def read(entry):
    path = fixture / entry['file']
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry['file_sha256']:
        raise ValueError('Fixture changed')
    return np.load(path, allow_pickle=False)
x = torch.from_numpy(read(record['input'])).to('cuda')
w = torch.from_numpy(read(record['weights']['weight'])).to(device='cuda', dtype=torch.float16)
bias = torch.from_numpy(read(record['weights']['bias'])).to(device='cuda', dtype=torch.float16)
out = torch.empty_like(x)
expected = read(record['output'])
cases = {}
try:
    for interval, rounding_mode in ((128, 0), (128, 3)):
        params = [V(x.data_ptr()), V(w.data_ptr()), V(bias.data_ptr()), V(out.data_ptr()),
                  I(512), I(1024), I(512), I(interval), I(rounding_mode)]
        argv = (V * len(params))(*[C.cast(C.byref(p), V).value for p in params])
        began = time.monotonic()
        check(launch(function, 32, 64, 1, 32, 1, 1, 0, V(torch.cuda.current_stream().cuda_stream), argv, None), 'launch')
        torch.cuda.synchronize()
        actual = out.cpu().numpy()
        diff = np.abs(actual.astype('f') - expected.astype('f'))
        key = f'{interval}_mode{rounding_mode}'
        cases[key] = {'sha256': hashlib.sha256(actual.tobytes()).hexdigest(),
                               'different_elements': int(np.count_nonzero(diff)), 'max_abs': float(diff.max()),
                               'mean_abs': float(diff.mean()), 'seconds': time.monotonic()-began}
        print('INTERVAL', key, json.dumps(cases[key]), flush=True)
finally:
    check(unload(module), 'unload')
report = {'device': torch.cuda.get_device_name(), 'visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
          'kernel_sha256': hashlib.sha256(source).hexdigest(), 'reference_sha256': record['output']['raw_sha256'],
          'scope': 'One latent projection only, not full model validation', 'cases': cases}
(root / f'projection_wmma_device{physical_device}_split_modes.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
