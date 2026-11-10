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
    output = pathlib.Path('importance-shards/operator_samples_remote.npz')
else:
    source = pathlib.Path('artifacts/unc/local_transform_diagnostic.npz')
    output = pathlib.Path('artifacts/unc/operator_samples_local.npz')
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
predict(m,xe)
np.savez_compressed(output,**samples)
print('OPERATOR_READY',len(samples),hashlib.sha256(output.read_bytes()).hexdigest(),flush=True)
