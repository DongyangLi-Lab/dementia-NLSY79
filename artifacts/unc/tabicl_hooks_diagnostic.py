import hashlib, json, pathlib, pickle, sys
import numpy as np
import torch
from inference import importance as imp, evaluation
from inference.scoring import predict

if sys.platform != 'win32':
    class Compatible(pickle.Unpickler):
        def find_class(self, module, name):
            return pathlib.PureWindowsPath if (module, name) == ('pathlib', 'WindowsPath') else super().find_class(module, name)
    def load(path):
        with open(path, 'rb') as f:
            return Compatible(f).load()
    imp.load_model = load
    source = pathlib.Path('importance-shards/local_transform_diagnostic.npz')
else:
    source = pathlib.Path('artifacts/unc/local_transform_diagnostic.npz')
z = np.load(source)
m, _, xe, y, ids, predictors = imp.load_temporal_model('TabICL', 6)
for cfg in (m.inference_config_.COL_CONFIG, m.inference_config_.ROW_CONFIG, m.inference_config_.ICL_CONFIG):
    cfg.verbose = True
xe = xe.iloc[:128]
for key, pp in m.ensemble_generator_.preprocessors_.items():
    a = z[key][:128]
    pp.transform = lambda x, a=a: a.copy() if len(x) == len(a) else (_ for _ in ()).throw(ValueError('unexpected shape'))
print('ENV', json.dumps({'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'gpu':torch.cuda.get_device_name(0),'cap':torch.cuda.get_device_capability(0),'amp':str(torch.get_autocast_dtype('cuda')),'tf32matmul':torch.backends.cuda.matmul.allow_tf32,'tf32cudnn':torch.backends.cudnn.allow_tf32,'matmul_precision':torch.get_float32_matmul_precision()}), flush=True)
def hook(name):
    def run(module, inputs, output):
        if isinstance(output, (tuple, list)): output = output[0]
        if not torch.is_tensor(output):
            print('LAYER',name,'not_tensor',type(output).__name__,flush=True); return
        a=output.detach().cpu().contiguous().numpy()
        print('LAYER',name,str(a.shape),str(a.dtype),hashlib.sha256(a.tobytes()).hexdigest(),float(np.mean(a,dtype=np.float64)),float(np.std(a,dtype=np.float64)),flush=True)
    return run
for name in ('col_embedder','row_interactor','icl_predictor'):
    getattr(m.model_,name).register_forward_hook(hook(name))
p = predict(m, xe)
saved = evaluation.load(6, 'TabICL', 'temporal').p.to_numpy()[:128]
d = np.abs(p-saved)
print('COMPARE',json.dumps({'max':float(d.max()),'mean':float(d.mean()),'passed':bool(np.allclose(p,saved,rtol=1e-5,atol=1e-7))}),flush=True)
