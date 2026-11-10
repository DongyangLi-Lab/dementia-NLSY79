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
    output = pathlib.Path('importance-shards/operator_samples_matrix_remote.npz')
else:
    source = pathlib.Path('artifacts/unc/local_transform_diagnostic.npz')
    output = pathlib.Path('artifacts/unc/operator_samples_matrix_local_v3.npz')
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

import os
seen_matmul = False
def inspect_matmul(module, inputs, output):
    global seen_matmul
    if seen_matmul: return
    seen_matmul=True
    x=inputs[0].detach(); w=module.weight.detach(); b=module.bias.detach() if module.bias is not None else None
    def report(name, t):
        a=t.detach().cpu().contiguous().numpy()
        print("MATRIX",name,str(a.shape),str(a.dtype),hashlib.sha256(a.tobytes()).hexdigest(),flush=True)
    print("MATRIX_ENV",sys.platform,torch.__version__,torch.version.cuda,torch.backends.cudnn.version(),torch.cuda.get_device_name(),"tf32",torch.backends.cuda.matmul.allow_tf32,"reduced",torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,"workspace",os.getenv("CUBLAS_WORKSPACE_CONFIG"),flush=True)
    report("input",x); report("weight",w); report("bias",b); report("actual",output)
    with torch.autocast(device_type="cuda", enabled=False):
        h=w.to(x.dtype); bias=b.to(x.dtype)
        report("half_linear",torch.nn.functional.linear(x,h,bias))
        report("half_mm_bias",(x.reshape(-1,x.shape[-1]) @ h.T + bias).reshape(output.shape))
        report("float_linear_half",torch.nn.functional.linear(x.float(),w.float(),b.float()).half())
        report("float_from_half",torch.nn.functional.linear(x.float(),h.float(),bias.float()).half())
        report("half_split",torch.cat([torch.nn.functional.linear(t,h,bias) for t in x.split(1,0)],0))
        for n in (2,4,8,16,32,47,64,128,256,512,1024,2048,4096):
            flat=x.reshape(-1,x.shape[-1]); report("chunk_"+str(n),torch.cat([torch.nn.functional.linear(t,h,bias) for t in flat.split(n,0)],0).reshape(output.shape))
        for n in (64,128,256,512,1024,2048):
            flat=x.reshape(-1,x.shape[-1]); pad=(-flat.shape[0])%n
            p=torch.nn.functional.pad(flat,(0,0,0,pad)); report("pad_"+str(n),torch.nn.functional.linear(p,h,bias)[:flat.shape[0]].reshape(output.shape))


mods['tf_col.blocks.0.multihead_attn1.linear2'].register_forward_hook(inspect_matmul)

predict(m,xe)
np.savez_compressed(output,**samples)
print('OPERATOR_READY',len(samples),hashlib.sha256(output.read_bytes()).hexdigest(),flush=True)
