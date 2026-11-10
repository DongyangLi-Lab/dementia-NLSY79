import os,sys,pathlib,pickle,hashlib,json,csv
import numpy as np
from scipy.stats import spearmanr
from inference import importance as imp
class U(pickle.Unpickler):
 def find_class(self,m,n): return pathlib.PureWindowsPath if (m,n)==('pathlib','WindowsPath') else super().find_class(m,n)
def load(p):
 with open(p,'rb') as f:return U(f).load()
imp.load_model=lambda p: None  # CPU analysis only: checkpoint hash checked, model not needed
root=pathlib.Path('importance-shards/shap-controls-011653')
manifest=json.loads((root/'manifest.json').read_text());ref=np.load(root/'sequential_reference.npz',allow_pickle=False)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
from artifact_store import job_dir
for name,path in {'model':job_dir('checkpoints',6,'final','TabICL')/'temporal.pkl','inference':job_dir('predictions',6,'final','TabICL')/'inference.done.json','importance':pathlib.Path(imp.__file__)}.items():
 assert sha(path)==manifest['core_stamp'][name],name
model,xd,xe,y,ids,P=imp.load_temporal_model('TabICL',6)
names=list(imp.predictor_groups(P)); assert len(names)==60
rng=np.random.default_rng(2026); rng.choice(len(xe),500,replace=False);rng.choice(len(xd),10,replace=False)
sizes=np.arange(1,60);w=59/(sizes*(60-sizes));Z=np.zeros((800,60),bool)
for i in range(0,800,2):
 Z[i,rng.choice(60,rng.choice(sizes,p=w/w.sum()),replace=False)]=True;Z[i+1]=~Z[i]
assert hashlib.sha256(np.ascontiguousarray(Z).tobytes()).hexdigest()=='6afb453cf62cbc975e3037abed8334024e287d03a13bc79ba1e8523fea2513a9'
remote=[np.load(root/('l40_diagnostic_'+n+'.npy'),allow_pickle=False) for n in ['background','explain','subject_000','subject_001']]
local=[ref['p0'],ref['p1'],ref['predictions'][0].reshape(-1),ref['predictions'][1].reshape(-1)]
assert [a.shape for a in remote]==[a.shape for a in local]==[(10,),(500,),(8000,),(8000,)]
def distribution(a,b):
 d=np.abs(a.astype(float)-b.astype(float)); r=d/np.abs(b.astype(float));
 return {'n':len(d),'abs_mean':float(d.mean()),'abs_p95':float(np.quantile(d,.95)),'abs_max':float(d.max()),'relative_mean':float(r.mean()),'relative_p50':float(np.quantile(r,.5)),'relative_p95':float(np.quantile(r,.95)),'relative_max':float(r.max()),'zero_reference_count':int(np.sum(b==0))}
def solve(arr):
 logit=lambda p:np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))
 v0=logit(arr[0].mean());v1=logit(arr[1][:2]);V=logit(np.stack([arr[2].reshape(800,10),arr[3].reshape(800,10)]).mean(axis=2)).T
 z=Z.astype(float);D=z[:,:-1]-z[:,[-1]];Y=(V-v0)-z[:,[-1]]*(v1-v0)[None,:];phi=np.linalg.lstsq(D,Y,rcond=None)[0].T
 return np.column_stack([phi,(v1-v0)-phi.sum(axis=1)])
a=solve(local);b=solve(remote);avg_a=np.mean(np.abs(a),axis=0);avg_b=np.mean(np.abs(b),axis=0)
top_a=np.argsort(-avg_a,kind='stable')[:10];top_b=np.argsort(-avg_b,kind='stable')[:10]
result={'purpose':'two-subject diagnostic; no production approval implied','coalition_array_sha256':hashlib.sha256(Z.tobytes()).hexdigest(),'core_stamp':manifest['core_stamp'],'files_sha256':{str(p):sha(p) for p in [root/'manifest.json',root/'sequential_reference.npz',*[root/('l40_diagnostic_'+n+'.npy') for n in ['background','explain','subject_000','subject_001']],pathlib.Path('icl_shap_controls_candidate_diag.py'),pathlib.Path('icl-shap-controls-8754716.log'),pathlib.Path('shard_importance_unc.py'),pathlib.Path('icl_cublaslt_split_rows.so')]},'execution':{'job':8754716,'gpu':'L40S g181103','transform':'fixed local CPU preprocess','algorithm':'3 column linear2 chunk 2048, ICL splitK4 algorithm21','batch_8000':{'col':112,'row':16330,'icl':22,'col_offload':True},'dtype':'reference and remote prediction arrays preserved during original mean/logit; constrained lstsq float64','seed':2026,'n_coalitions':800,'background':10,'shap_unit':'log-odds'},'probability':{n:distribution(r,l) for n,r,l in zip(['p0','p1','subject0','subject1'],remote,local)},'subjects':[],'two_subject_mean_absolute':{'spearman':float(spearmanr(avg_a,avg_b).statistic),'top10_overlap':int(len(set(top_a)&set(top_b))),'local_top10':[names[i] for i in top_a],'remote_top10':[names[i] for i in top_b]}}
for k in range(2):
 diff=np.abs(a[k]-b[k]);top=np.argsort(-np.abs(a[k]),kind='stable')[:10]
 result['subjects'].append({'subject':k,'shap_abs_difference_max':float(diff.max()),'shap_abs_difference_mean':float(diff.mean()),'shap_spearman':float(spearmanr(a[k],b[k]).statistic),'top10_local_abs_direction_agreement':int(np.sum(np.sign(a[k,top])==np.sign(b[k,top]))),'top10_local_abs':[names[i] for i in top]})
out=root/'impact_8754716.json';out.write_text(json.dumps(result,indent=2,ensure_ascii=False))
with (root/'impact_8754716.csv').open('w',newline='') as f:
 wr=csv.writer(f);wr.writerow(['feature','subject0_local','subject0_l40s','subject1_local','subject1_l40s','mean_abs_local','mean_abs_l40s']);wr.writerows([[names[i],a[0,i],b[0,i],a[1,i],b[1,i],avg_a[i],avg_b[i]] for i in range(60)])
print(json.dumps(result,indent=2));print('OUTPUT_SHA',sha(out),sha(root/'impact_8754716.csv'))
