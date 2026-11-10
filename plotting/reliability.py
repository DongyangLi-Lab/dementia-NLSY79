"""Paired household uncertainty for calibration and decision curves of local refits."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
R=Path();S=Path()
M=['LR','GNB','DT','LightGBM','CatBoost','RealMLP','GAT','TabICL']
T=np.round(np.arange(.01,.40001,.005),6)
B=2000;SEED=20261022

def load(h):
    ts=[pd.read_csv(S/f'h{h}'/m/'temporal_predictions.csv') for m in M]
    ids=ts[0].hhidpn.to_numpy(dtype=np.int64);y=ts[0].y.to_numpy()
    for t in ts:assert np.array_equal(t.hhidpn,ids) and np.array_equal(t.y,y)
    assert len(np.unique(ids))==len(ids) and np.isin(y,[0,1]).all()
    data=[]
    for t in ts:
        p=t.p.to_numpy(); assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
        u,inv=np.unique(p,return_inverse=True);inv=len(u)-1-inv
        j=np.searchsorted(-u[::-1],-T,side='right')-1
        edges=np.unique(np.quantile(p,np.linspace(0,1,11)))
        # Discrete predictions can leave tiny quantile bins; merge adjacent bins,
        # retaining every participant, before defining conditional-bin intervals.
        while len(edges)>2:
            codes=np.searchsorted(edges[1:-1],p,side='right')
            sizes=np.bincount(codes,minlength=len(edges)-1)
            small=np.flatnonzero(sizes<100)
            if not len(small):break
            z=int(small[0]);boundary=z+1 if z<len(sizes)-1 else z
            edges=np.delete(edges,boundary)
        bins=np.searchsorted(edges[1:-1],p,side='right')
        data.append(dict(p=p,inv=inv,ng=len(u),j=j,bins=bins,nb=int(bins.max()+1),edges=edges))
    return dict(ids=ids,hh=ids//1000,y=y,n=len(y),models=data)

def evaluate(d,c):
    n=c.sum();y=d['y'];prevalence=c@y/n
    all_reference=prevalence-(1-prevalence)*T/(1-T)
    nb=[];cal=[]
    for v in d['models']:
        group=np.bincount(v['inv'],weights=c,minlength=v['ng'])
        cases=np.bincount(v['inv'],weights=c*y,minlength=v['ng'])
        cc=np.cumsum(cases);nn=np.cumsum(group)
        tp=np.where(v['j']>=0,cc[np.maximum(v['j'],0)],0)
        selected=np.where(v['j']>=0,nn[np.maximum(v['j'],0)],0)
        nb.append(tp/n-(selected-tp)/n*T/(1-T))
        mass=np.bincount(v['bins'],weights=c,minlength=v['nb'])
        ps=np.bincount(v['bins'],weights=c*v['p'],minlength=v['nb'])
        ys=np.bincount(v['bins'],weights=c*y,minlength=v['nb'])
        assert np.all(mass>0)
        cal.append(np.stack([ps/mass,ys/mass,(ys-ps)/mass],axis=1))
    nb=np.asarray(nb)
    return nb,nb-np.maximum(all_reference,0),all_reference,cal

def run(result, output, samples=2000):
    global R,S,B
    R=Path(output);S=Path(result)/"predictions/final";B=samples
    ds={h:load(h) for h in [6,8]};hh=np.union1d(ds[6]['hh'],ds[8]['hh'])
    for d in ds.values():d['code']=np.searchsorted(hh,d['hh'])
    points={h:evaluate(d,np.ones(d['n'])) for h,d in ds.items()}
    boots={h:np.empty((B,8,len(T))) for h in ds}
    above={h:np.empty((B,8,len(T))) for h in ds}
    cal={h:[np.empty((B,v['nb'],3)) for v in d['models']] for h,d in ds.items()}
    rng=np.random.default_rng(SEED)
    for b in range(B):
        c=np.bincount(rng.integers(0,len(hh),len(hh)),minlength=len(hh))
        for h,d in ds.items():
            nb,a,ta,k=evaluate(d,c[d['code']]);boots[h][b]=nb;above[h][b]=a
            for j in range(8):cal[h][j][b]=k[j]
        if (b+1)%500==0:print(f'{b+1}/{B}',flush=True)
    rows=[];cr=[];diff=[]
    for h,d in ds.items():
        nb,a,ta,k=points[h]
        for j,m in enumerate(M):
            lo,hi=np.percentile(boots[h][:,j,:],[2.5,97.5],axis=0)
            al,ah=np.percentile(above[h][:,j,:],[2.5,97.5],axis=0)
            for z,t in enumerate(T):rows.append(dict(horizon=h,model=m,threshold=t,n=d['n'],net_benefit=nb[j,z],lo=lo[z],hi=hi[z],above_default=a[j,z],above_lo=al[z],above_hi=ah[z],treat_all=ta[z],treat_none=0))
            cl,ch=np.percentile(cal[h][j],[2.5,97.5],axis=0)
            for z in range(len(k[j])):
                entry=dict(horizon=h,model=m,bin=z,n=int(np.sum(d['models'][j]['bins']==z)),n_total=d['n'])
                for w,name in enumerate(['predicted','observed','residual']):entry.update({name:k[j][z,w],name+'_lo':cl[z,w],name+'_hi':ch[z,w]})
                cr.append(entry)
        for model,reference in [(m,'LR') for m in M[1:]]+[('TabICL','CatBoost')]:
            i=M.index(model);j=M.index(reference)
            v=nb[i]-nb[j];lo,hi=np.percentile(boots[h][:,i,:]-boots[h][:,j,:],[2.5,97.5],axis=0)
            for z,t in enumerate(T):diff.append(dict(horizon=h,model=model,reference=reference,threshold=t,difference=v[z],lo=lo[z],hi=hi[z]))
    pd.DataFrame(rows).to_csv(R/'decision_ci.csv',index=False)
    pd.DataFrame(cr).to_csv(R/'calibration_ci.csv',index=False)
    pd.DataFrame(diff).to_csv(R/'decision_difference_ci.csv',index=False)
    # Independent point-estimate calculation at representative thresholds.
    for h,d in ds.items():
        for j,v in enumerate(d['models']):
            for t in [.05,.1,.2,.4]:
                z=np.flatnonzero(np.isclose(T,t))[0];sel=v['p']>=t
                exact=np.mean(sel*d['y'])-np.mean(sel*(1-d['y']))*t/(1-t)
                assert np.isclose(exact,points[h][0][j,z],atol=1e-12)
    # One literal duplicated-household sample verifies weighted computation and tie handling.
    d=ds[6];c=np.bincount(rng.integers(0,len(hh),len(hh)),minlength=len(hh))[d['code']]
    idx=np.repeat(np.arange(d['n']),c);weighted=evaluate(d,c)[0]
    for j,v in enumerate(d['models']):
        for t in [.05,.1,.2,.4]:
            z=np.flatnonzero(np.isclose(T,t))[0];sel=v['p'][idx]>=t;y=d['y'][idx]
            exact=np.mean(sel*y)-np.mean(sel*(1-y))*t/(1-t)
            assert np.isclose(exact,weighted[j,z],atol=1e-12)
    (R/'reliability_method.json').write_text(json.dumps(dict(n_boot=B,seed=SEED,models=M,unit='Household hhidpn // 1000',
        pairing='Shared household multiplicities across all models and overlapping horizons, sampled from union households',
        scope='Saved revised-outcome fitted models; no retraining',
        calibration='Fixed model-specific empirical quantile-bin boundaries from revised temporal cohort; duplicate boundaries collapsed and adjacent bins merged until all have at least 100 original participants. No participants excluded. CIs conditional on model and these boundaries. Bootstrap residual recomputes both observed and mean predicted risk.',
        net_benefit='TP/n - FP/n * threshold/(1-threshold); threshold fixed at each grid point, no outcome-based threshold selection',
        intervals='Pointwise percentile 95% CIs, no simultaneous/multiplicity coverage claimed',
        context='Exploratory threshold analysis, no validated clinical action or intervention benefit',
        source_sha256={str(p.relative_to(S)):hashlib.sha256(p.read_bytes()).hexdigest() for p in S.glob('h*/*/temporal_predictions.csv')},
        point_formula_verified=True,weighted_vs_literal_household_duplication_verified=True),indent=2),encoding='utf8')
    print('Saved paired calibration and decision-curve intervals',flush=True)
