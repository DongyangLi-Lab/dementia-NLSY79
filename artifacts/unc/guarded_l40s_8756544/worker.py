"""Guarded, resumable L40S KernelSHAP with per-execution warmups and evidence."""
import os, sys, json, hashlib, pathlib, pickle, ctypes, types, time, subprocess
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from inference import importance as imp, evaluation
from inference.scoring import predict
from artifact_store import job_dir
from tabicl.model.inference import InferenceManager

BASE = pathlib.Path('importance-shards')
INPUT = BASE / 'production-inputs-20261110-013329'
CONTROL = BASE / 'shap-controls-011653'
DECISION = BASE / 'tabicl_cross_device_acceptance.json'
DESIGN = BASE / 'tabicl_kernel_shap_design.npz'
SO = pathlib.Path('icl_cublaslt_split_rows.so')
SOURCE = pathlib.Path(__file__).resolve()
os.umask(0o077)
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for part in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(part)
    return h.hexdigest()
def array_hash(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def require(ok, msg):
    if not ok: raise RuntimeError(msg)
def save_json(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)); os.replace(tmp, path)
def save_npz(path, **items):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'wb') as f: np.savez_compressed(f, **items)
    os.replace(tmp, path)
class U(pickle.Unpickler):
    def find_class(self, module, name):
        return pathlib.PureWindowsPath if (module, name) == ('pathlib','WindowsPath') else super().find_class(module,name)
def load(p):
    with open(p,'rb') as f: return U(f).load()
imp.load_model = load
require(sha(DECISION)=='fe13eadbe68dbd5ae491cfedd814fcd1a179c707e7c64a61c6a7a54ffd3667d4','acceptance hash')
decision=json.loads(DECISION.read_text()); core=decision['evidence']['core_stamp']
require(decision['status']=='accepted_for_guarded_production','decision status')
require(sha(INPUT/'manifest.json')==decision['production_input_manifest_sha256'],'cache manifest hash')
manifest=json.loads((INPUT/'manifest.json').read_text())
require(manifest['core_stamp']==core,'cache core stamp')
paths={'model':job_dir('checkpoints',6,'final','TabICL')/'temporal.pkl', 'inference':job_dir('predictions',6,'final','TabICL')/'inference.done.json', 'importance':pathlib.Path(imp.__file__), 'design':DESIGN, 'worker':BASE/'tabicl_kernel_shap_sharded.py', 'exclusive_guard':BASE/'audit_tabicl_permutations.py'}
for k,p in paths.items(): require(sha(p)==core[k], 'core hash '+k)
require(sha(SO)=='ffcb1ed323ad164a840d54efd3d8488489ec5979825597d94921b8e13e7f92f8','library hash')
require(sha(CONTROL/'sequential_reference.npz')=='4d710b14c7c2c5aecc51fae2317f0b914cd3a646ca0db39539ce6c5e78e198a5','reference hash')
design=np.load(DESIGN,allow_pickle=False);ref=np.load(CONTROL/'sequential_reference.npz',allow_pickle=False)
z=design['coalitions'].astype(float); mask=design['column_mask']; cols=[str(x) for x in design['columns']]
require(z.shape==(800,60) and mask.shape==(800,94),'design shape')
require(array_hash(design['coalitions'])=='6afb453cf62cbc975e3037abed8334024e287d03a13bc79ba1e8523fea2513a9','coalition content')
require(torch.cuda.is_available() and 'L40S' in torch.cuda.get_device_name(0),'wrong GPU')
torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(True);torch.backends.cuda.enable_math_sdp(False);torch.backends.cuda.enable_cudnn_sdp(False)
model,xd,xe,y,ids,P=imp.load_temporal_model('TabICL',6)
require(cols==list(xe.columns)==manifest['columns'],'columns')
require([str(x) for x in design['groups']]==list(imp.predictor_groups(P)),'groups')
for key,pp in model.ensemble_generator_.preprocessors_.items(): require(array_hash(pp.X_transformed_)==manifest['training_transforms'][key]['sha256'],'training transform '+key)
bg=xd.to_numpy(np.float32)[design['background_indices']]; xeall=xe.to_numpy(np.float32)[design['explain_indices']]
saved=evaluation.load(6,'TabICL','temporal'); require(np.array_equal(y,saved.y.to_numpy()) and np.array_equal(ids.astype(str),saved.hhidpn.astype(str)),'baseline identity')
idx=int(os.environ.get('SLURM_ARRAY_TASK_ID','0')); mode=os.environ.get('SHAP_MODE','production')
require(mode in ('production','validate'),'mode'); require(mode=='validate' or 1<=idx<=19,'ownership')
job=os.environ.get('SLURM_JOB_ID','unknown'); run=BASE/'guarded-l40s-production'/('validate-'+job if mode=='validate' else 'shard-%02d-job-%s'%(idx,job)); run.mkdir(parents=True,exist_ok=False)
free,total=torch.cuda.mem_get_info(); reservation=torch.empty(max(0,free-15*1024**3),dtype=torch.uint8,device='cuda')
for i in range(3):
    layer=dict(model.model_.col_embedder.named_modules())['tf_col.blocks.%d.multihead_attn1.linear2'%i]
    def chunk_forward(self,x):
        a=x.reshape(-1,x.shape[-1]);out=torch.cat([F.linear(t,self.weight,self.bias) for t in a.split(2048,0)],0);return out.reshape(*x.shape[:-1],self.out_features)
    layer.forward=types.MethodType(chunk_forward,layer)
lib=ctypes.CDLL(str(SO.resolve()));fn=lib.try_split_rows;fn.argtypes=[ctypes.c_void_p]*5+[ctypes.c_size_t,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int)];fn.restype=ctypes.c_int
workspace=torch.empty(512*1024*1024,dtype=torch.uint8,device='cuda'); original_linear=F.linear;weights={id(b.attn.out_proj.weight) for b in model.model_.icl_predictor.tf_icl.blocks};state={'rows':None,'calls':0,'batch':{},'offload':{},'full':False};seen={}
def replacement(x,w,b=None):
    if id(w) not in weights or x.ndim!=3 or x.shape[-2]!=state['rows']: return original_linear(x,w,b)
    require(x.dtype==torch.float16 and b is not None,'candidate dtype');xx=x.contiguous();ww=w.to(x.dtype).contiguous();bb=b.to(x.dtype).contiguous();out=torch.empty_like(xx);count=ctypes.c_int();picked=ctypes.c_int();stream=torch.cuda.current_stream().cuda_stream;rows=xx.numel()//xx.shape[-1];found=False
    for i in range(16):
        status=fn(xx.data_ptr(),ww.data_ptr(),bb.data_ptr(),out.data_ptr(),workspace.data_ptr(),workspace.numel(),stream,i,4,1,rows,ctypes.byref(count),ctypes.byref(picked))
        if status==0 and picked.value==21: found=True;break
        if i+1>=count.value: break
    require(found,'splitK algorithm21 unsupported');state['calls']+=1;return out
old_est=InferenceManager.estimate_safe_batch_size; old_call=InferenceManager.__call__;old_split=InferenceManager.compute_split_sizes
def estimate(self,*args,**kw):
    mem,n=old_est(self,*args,**kw)
    if state['full']: n={'tf_col':88,'tf_row':16456,'tf_icl':17}[self.enc_name]
    state['batch'].setdefault(self.enc_name,[]).append(int(n));return mem,n
def call(self,*args,**kw):
    if state['full']: self.offload=(self.enc_name=='tf_col')
    out=old_call(self,*args,**kw);state['offload'].setdefault(self.enc_name,[]).append(bool(self.offload));return out
def split(dims,batch):
    key=tuple(dims)
    if state['full']:
        targets={(98,):88,(8,27369):16456}
        require(key in targets and batch==targets[key], 'full batch fallback '+str((key,batch)))
    if key in seen: require(seen[key]==batch,'batch fallback')
    seen[key]=batch;return old_split(dims,batch)
InferenceManager.estimate_safe_batch_size=estimate;InferenceManager.__call__=call;InferenceManager.compute_split_sizes=staticmethod(split);F.linear=replacement
def get_raw(name):
    if name=='full': return xe.to_numpy(np.float32)
    if name=='background': return bg
    if name=='explain': return xeall
    i=int(name[-3:]);return np.where(mask[:,None,:],xeall[i][None,None,:],bg[None,:,:]).reshape(-1,len(cols))
def run_case(name):
    raw=get_raw(name);meta=manifest['cases'][name];require(array_hash(raw)==meta['input']['sha256'],'raw input '+name);p=INPUT/meta['file'];require(sha(p)==meta['file_sha256'],'cache file '+name)
    cache=np.load(p,allow_pickle=False)
    for key,pp in model.ensemble_generator_.preprocessors_.items():
        arr=cache[key];require(array_hash(arr)==meta['transforms'][key]['sha256'],'transform '+name+'/'+key)
        pp.transform=lambda x,arr=arr: arr.copy() if len(x)==len(arr) else (_ for _ in ()).throw(ValueError('cache length'))
    state.update(rows=len(raw),calls=0,batch={},offload={},full=name=='full');seen.clear();t=time.monotonic();out=np.asarray(predict(model,pd.DataFrame(raw,columns=cols)))
    require(state['calls']==24,'projection count');require(np.isfinite(out).all() and np.all((out>=0)&(out<=1)),'invalid probability')
    route={'batch':state['batch'].copy(),'offload':state['offload'].copy(),'calls':state['calls'],'seconds':time.monotonic()-t,'input_sha256':meta['input']['sha256'],'cache_file_sha256':meta['file_sha256'],'transform_hashes':{k:v['sha256'] for k,v in meta['transforms'].items()}}
    if name.startswith('subject_'):
        require(all(x==112 for x in route['batch']['tf_col']) and all(x==16330 for x in route['batch']['tf_row']) and all(x==22 for x in route['batch']['tf_icl']), 'unexpected 8000-row batch '+str(route))
        require(all(route['offload']['tf_col']) and not any(route['offload']['tf_row']+route['offload']['tf_icl']),'unexpected offload')
    print('CASE',name,json.dumps(route),flush=True);return out,route
def phi(p0,p1,ps):
    logit=lambda p:np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))
    v0=logit(p0.mean());v1=logit(p1[:2]);V=logit(np.stack([p.reshape(800,10) for p in ps]).mean(axis=2)).T;D=z[:,:-1]-z[:,[-1]];Y=(V-v0)-z[:,[-1]]*(v1-v0)[None,:];p=np.linalg.lstsq(D,Y,rcond=None)[0].T
    return np.column_stack([p,(v1-v0)-p.sum(axis=1)])
routes={};full,routes['full']=run_case('full');d=np.abs(full-saved.p.to_numpy()); report={'full_max_abs':float(d.max()),'full_strict_allclose':bool(np.allclose(full,saved.p.to_numpy(),rtol=1e-5,atol=1e-7))}
require(report['full_max_abs']<=.001,'full engineering gate'); del full;torch.cuda.empty_cache()
p0,routes['background']=run_case('background');p1,routes['explain']=run_case('explain');ps=[]
for i in range(2):
    p,routes['subject_%03d'%i]=run_case('subject_%03d'%i);ps.append(p)
local=[ref['p0'],ref['p1'],ref['predictions'][0].reshape(-1),ref['predictions'][1].reshape(-1)];remote=[p0,p1,*ps];report['probability_max_abs']=[float(np.max(np.abs(a-b))) for a,b in zip(remote,local)];report['original_strict_allclose']=[bool(np.allclose(a,b,rtol=1e-5,atol=1e-7)) for a,b in zip(remote,local)];require(max(report['probability_max_abs'])<=.0005,'control probability gate')
a=phi(local[0],local[1],local[2:]);b=phi(p0,p1,ps);report['individual']=[]
for i in range(2):
    top=np.argsort(-np.abs(a[i]),kind='stable')[:10];item={'max_abs':float(np.max(np.abs(a[i]-b[i]))),'relative_l1':float(np.sum(np.abs(a[i]-b[i]))/np.sum(np.abs(a[i]))),'rho':float(spearmanr(a[i],b[i]).statistic),'direction_agreement':int(np.sum(np.sign(a[i,top])==np.sign(b[i,top])))};report['individual'].append(item)
    require(item['max_abs']<=.001 and item['relative_l1']<=.01 and item['rho']>=.99 and item['direction_agreement']==10,'individual SHAP gate')
aa=np.mean(np.abs(a),axis=0);bb=np.mean(np.abs(b),axis=0);report['mean_abs_rho']=float(spearmanr(aa,bb).statistic);report['top10_overlap']=len(set(np.argsort(-aa)[:10])&set(np.argsort(-bb)[:10]));require(report['mean_abs_rho']>=.99 and report['top10_overlap']>=9,'aggregate SHAP gate')
report.update(status='passed_engineering_gate',interpretation=decision['interpretation'],routes=routes);save_json(run/'startup_checks.json',report);save_npz(run/'startup_predictions.npz',p0=p0,p1=p1,predictions=np.stack(ps).reshape(2,800,10));print('STARTUP_PASSED',json.dumps(report),flush=True)
indices=[] if mode=='validate' else list(range(idx*25,idx*25+25));out={'status':'running','indices':indices,'completed':{},'core_stamp':core,'acceptance_sha256':sha(DECISION),'input_manifest_sha256':sha(INPUT/'manifest.json'),'startup_report_sha256':sha(run/'startup_checks.json'),'startup_predictions_sha256':sha(run/'startup_predictions.npz'),'worker_sha256':sha(SOURCE),'patch_library_sha256':sha(SO),'patch_source_sha256':sha('importance-shards/icl_cublaslt_split.cc'),'diagnostic_source_sha256':sha('icl_shap_controls_candidate_diag.py'),'job_id':job,'array_index':idx,'gpu':torch.cuda.get_device_name(0),'gpu_total_bytes':total,'memory_reservation_bytes':int(max(0,free-15*1024**3)),'torch_version':torch.__version__,'cuda_version':torch.version.cuda,'routes':routes,'strict_gate':'failed; engineering acceptance only'};save_json(run/'run_manifest.json',out)
for i in indices:
    name='subject_%03d'%i;dest=run/(name+'.npz');require(not dest.exists(),'refusing overwrite');p,route=run_case(name);save_npz(dest,p0=p0,p1=p1,predictions=p.reshape(1,800,10));out['completed'][str(i)]={'file':dest.name,'sha256':sha(dest),'route':route};save_json(run/'run_manifest.json',out);print('SUBJECT_SAVED',i,sha(dest),flush=True)
out['status']='complete';save_json(run/'run_manifest.json',out);print('RUN_COMPLETE',run,flush=True)
