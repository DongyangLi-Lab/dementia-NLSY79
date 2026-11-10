import hashlib, pathlib, pickle, sys
import numpy as np
import torch
from inference import importance as imp
from inference.scoring import predict
if sys.platform != 'win32':
    class Compatible(pickle.Unpickler):
        def find_class(self, module, name):
            return pathlib.PureWindowsPath if (module, name) == ('pathlib', 'WindowsPath') else super().find_class(module, name)
    def load(path):
        with open(path, 'rb') as f: return Compatible(f).load()
    imp.load_model = load
    source = pathlib.Path('importance-shards/local_transform_diagnostic.npz')
    output = pathlib.Path('importance-shards/operator_samples_icl_full_reference_remote.npz')
else:
    source = pathlib.Path('artifacts/unc/local_transform_diagnostic.npz')
    output = pathlib.Path('artifacts/unc/operator_samples_icl_projection_capture_local.npz')
assert hashlib.sha256(source.read_bytes()).hexdigest() == '45d87bc03f17186a65feb4d0acfe79ebc768a1b882cae5689f20c3ea5d85680d'
z = np.load(source)
m, _, xe, _, _, _ = imp.load_temporal_model('TabICL', 6)
xe = xe.iloc[:128]
for key, pp in m.ensemble_generator_.preprocessors_.items():
    a = z[key][:128]
    pp.transform = lambda x, a=a: a.copy() if len(x)==len(a) else (_ for _ in ()).throw(ValueError('unexpected shape'))
samples={}
def hook(name):
    def f(module, inputs, out):
        if name in samples: return
        if isinstance(out,(tuple,list)): out=out[0]
        if not torch.is_tensor(out): return
        t=out.detach().reshape(-1)
        count=min(t.numel(),8192)
        ix=torch.arange(count,device=t.device,dtype=torch.int64)*(t.numel()-1)//max(1,count-1)
        a=t[ix].cpu().numpy(); samples[name]=a
        print('OPERATOR',name,'shape',tuple(out.shape),'dtype',str(out.dtype),'sample_sha256',hashlib.sha256(a.tobytes()).hexdigest(),flush=True)
    return f
mods=dict(m.model_.col_embedder.named_modules())
keys=['in_linear','tf_col.blocks.0.multihead_attn1.norm1','tf_col.blocks.0.multihead_attn1.attn','tf_col.blocks.0.multihead_attn1.linear1','tf_col.blocks.0.multihead_attn1.linear2','tf_col.blocks.0.multihead_attn1','tf_col.blocks.0.multihead_attn2','tf_col.blocks.0']
for key in keys: mods[key].register_forward_hook(hook(key))
for key, mod in mods.items():
    if key.startswith('tf_col.blocks.') and key.count('.')==2: mod.register_forward_hook(hook('BLOCK_'+key))
for key in ('col_embedder','row_interactor','icl_predictor'):
    getattr(m.model_,key).register_forward_hook(hook('STAGE_'+key))
for key,mod in m.model_.icl_predictor.named_modules():
    if (key.startswith('tf_icl.blocks.0.') and key.count('.')==3) or (key.startswith('tf_icl.blocks.') and key.count('.')==2) or key in ('y_encoder','ln','decoder'):
        mod.register_forward_hook(hook('ICL_'+key))
def full_hash(name):
    def h(module, args, output=None):
        if name in samples: return
        value=args[0] if output is None else output
        a=value.detach().cpu().contiguous().numpy()
        samples[name]=np.array([0],dtype=np.uint8)
        print('FULL',name,str(a.shape),str(a.dtype),hashlib.sha256(a.tobytes()).hexdigest(),flush=True)
    return h
mods['tf_col.blocks.0.multihead_attn1.linear1'].register_forward_hook(full_hash('linear1_output'))
mods['tf_col.blocks.0.multihead_attn1.linear2'].register_forward_pre_hook(full_hash('linear2_input'))

# Run only during a coordinated exclusive GPU window.
import os, types
import torch.nn.functional as F
attn0 = m.model_.icl_predictor.tf_icl.blocks[0].attn
original_attn = attn0.forward
original_projection = F._in_projection_packed
original_sdpa = F.scaled_dot_product_attention
original_linear = F.linear
captured = False
def tensor_hash(name, t):
    a = t.detach().cpu().contiguous().numpy()
    print('ICL_FULL', name, tuple(a.shape), str(a.dtype), hashlib.sha256(a.tobytes()).hexdigest(), flush=True)
def inspect_forward(self, *args, **kwargs):
    global captured
    if captured: return original_attn(*args, **kwargs)
    captured = True
    print('REFERENCE_GPU', torch.cuda.get_device_name(), 'CUDA_VISIBLE_DEVICES', os.getenv('CUDA_VISIBLE_DEVICES'), 'MEM_INFO', torch.cuda.mem_get_info(), flush=True)
    for n, x in zip(('query', 'key', 'value'), args[:3]): tensor_hash('input_'+n, x)
    counts = {'sdpa': 0, 'out': 0}
    def inspect_projection(*a, **kw):
        r = original_projection(*a, **kw)
        for n, x in zip(('q', 'k', 'v'), r): tensor_hash('projected_'+n, x)
        return r
    def inspect_sdpa(*a, **kw):
        r = original_sdpa(*a, **kw); counts['sdpa'] += 1
        tensor_hash('sdpa_output_'+str(counts['sdpa']), r); return r
    def inspect_linear(x, w, b=None):
        r = original_linear(x, w, b)
        if w is self.out_proj.weight:
            counts['out'] += 1; tensor_hash('out_projection_'+str(counts['out']), r)
            if counts['out'] == 2:
                replay = pathlib.Path('artifacts/unc/icl_query_projection_input_private.npz')
                with replay.open('xb') as f:
                    np.savez_compressed(f, x=x.detach().cpu().numpy(), weight=w.detach().cpu().numpy(), bias=b.detach().cpu().numpy(), output=r.detach().cpu().numpy())
                print('PROJECTION_CAPTURE',str(replay),hashlib.sha256(replay.read_bytes()).hexdigest(),flush=True)
        return r
    F._in_projection_packed = inspect_projection
    F.scaled_dot_product_attention = inspect_sdpa
    F.linear = inspect_linear
    try:
        result = original_attn(*args, **kwargs)
        tensor_hash('attention_output', result)
        return result
    finally:
        F._in_projection_packed = original_projection
        F.scaled_dot_product_attention = original_sdpa
        F.linear = original_linear
attn0.forward = types.MethodType(inspect_forward, attn0)
print('ICL_REFERENCE_EXCLUSIVE_WINDOW_REQUIRED', flush=True)

predict(m,xe)
np.savez_compressed(output,**samples)
print('OPERATOR_READY',len(samples),hashlib.sha256(output.read_bytes()).hexdigest(),flush=True)
