import pathlib, pickle, sys
import numpy as np
import torch
from inference import importance as imp
from inference.scoring import predict
if sys.platform != 'win32':
    class Compatible(pickle.Unpickler):
        def find_class(self, module, name):
            return pathlib.PureWindowsPath if (module, name) == ('pathlib', 'WindowsPath') else super().find_class(module, name)
    def load(path):
        with open(path,'rb') as f: return Compatible(f).load()
    imp.load_model=load
    source=pathlib.Path('importance-shards/local_transform_diagnostic.npz')
else:
    source=pathlib.Path('artifacts/unc/local_transform_diagnostic.npz')
z=np.load(source)
m,_,xe,_,_,_=imp.load_temporal_model('TabICL',6)
xe=xe.iloc[:128]
for key,pp in m.ensemble_generator_.preprocessors_.items():
    a=z[key][:128]
    pp.transform=lambda x,a=a: a.copy() if len(x)==len(a) else (_ for _ in ()).throw(ValueError('shape'))
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p:
    predict(m,xe)
for x in p.key_averages():
    k=x.key.lower()
    if any(s in k for s in ('scaled_dot_product','flash_attention','efficient_attention','cudnn_attention','bmm','matmul')):
        print('PROFILE',x.key,'calls',x.count,flush=True)
