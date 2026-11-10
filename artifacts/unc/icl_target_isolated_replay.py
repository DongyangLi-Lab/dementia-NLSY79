"""Run only on the approved UNC GPU after private replay data are authorized and staged."""
import argparse, ctypes, hashlib, json, os
from pathlib import Path
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True)
parser.add_argument('--lib', required=True)
args = parser.parse_args()
reference_hash = 'a81e0af7f618308d7ef653513d89fd808280a103c765cb38a64499ad4d226eca'
def digest(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
with np.load(args.input, allow_pickle=False) as z:
    x, weight, bias, expected = (z[k].copy() for k in ('x','weight','bias','output'))
assert x.shape == (8,128,512) and x.dtype == np.float16
assert weight.shape == (512,512) and bias.shape == (512,)
assert expected.shape == x.shape and expected.dtype == np.float16
assert digest(expected) == reference_hash, 'isolated input is not paired with the approved local reference'
assert np.isfinite(x).all() and np.isfinite(weight).all() and np.isfinite(bias).all()
assert torch.cuda.is_available()
x = torch.from_numpy(x).cuda().contiguous()
w = torch.from_numpy(weight).to(device='cuda',dtype=x.dtype).contiguous()
b = torch.from_numpy(bias).to(device='cuda',dtype=x.dtype).contiguous()
out = torch.empty_like(x); work = torch.empty(32*1024*1024,dtype=torch.uint8,device='cuda')
lib=ctypes.CDLL(os.path.abspath(args.lib)); run=lib.try_target
run.argtypes=[ctypes.c_void_p]*5+[ctypes.c_size_t,ctypes.c_void_p,ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int)]
run.restype=ctypes.c_int
count=ctypes.c_int(); picked=ctypes.c_int(); stream=torch.cuda.current_stream().cuda_stream
print('ISOLATED_CONFIG',json.dumps({'gpu':torch.cuda.get_device_name(),'input_sha256':hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),'reference_sha256':reference_hash,'lib_sha256':hashlib.sha256(Path(args.lib).read_bytes()).hexdigest()}),flush=True)
for i in range(16):
    status=run(x.data_ptr(),w.data_ptr(),b.data_ptr(),out.data_ptr(),work.data_ptr(),work.numel(),stream,i,ctypes.byref(count),ctypes.byref(picked)); torch.cuda.synchronize()
    if status == 0:
        actual=out.detach().cpu().numpy(); delta=np.abs(actual.astype(np.float32)-expected.astype(np.float32)); h=digest(actual)
        print('ISOLATED_RESULT',json.dumps({'index':i,'algorithm_id':picked.value,'candidates':count.value,'sha256':h,'exact':h==reference_hash,'max_abs':float(delta.max()),'different_elements':int(np.count_nonzero(delta))}),flush=True)
    else: print('ISOLATED_RESULT',json.dumps({'index':i,'status':status,'candidates':count.value}),flush=True)
    if i+1>=count.value: break
print('ISOLATED_READY',flush=True)
