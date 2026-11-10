from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch,PathPatch,Rectangle
from matplotlib.path import Path as MPath
from scipy.spatial.distance import cdist
from scipy.stats import norm
from audit_panel_alignment import require_matplotlib_panel_alignment
from svg_compat import normalize

O=OUT=Path(__file__).resolve().parents[1]
REPO=Path();PREP=O/'input_tables';RESULT=Path();DPI=300
WINE='#A3133F';PURPLE='#553574';GOLD='#AD6500';INK='#25252B';GRAY='#96929A';LIGHT='#E6E1E8';BLUE='#1774CB'
NAV=['No difficulty','Difficulty','Does not use maps','Unrecorded'];NC=[PURPLE,WINE,GOLD,GRAY]
INPUTS=set();CHECKS=[];FINAL=False
plt.rcParams.update({'font.family':'Arial','font.size':7,'axes.labelsize':7.5,'axes.titlesize':8,
 'xtick.labelsize':6.5,'ytick.labelsize':6.5,'axes.linewidth':.85,'xtick.major.width':.7,'ytick.major.width':.7,
 'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none',
 'legend.frameon':False,'figure.facecolor':'white','savefig.facecolor':'white'})
def read(p):
 p=Path(p).resolve()
 if not (p.is_relative_to(RESULT.resolve()) or p.is_relative_to(O.resolve())):
  raise ValueError(f'Plotting input is outside result/output: {p}')
 INPUTS.add(p);return pd.read_csv(p)
def save(d,n):d.to_csv(O/'source_data'/f'{n}.csv',index=False,encoding='utf-8-sig')
def check(n,t,detail=None):
 assert bool(t),(n,detail)
 CHECKS.append({'check':n,'passed':True,'detail':detail})
def letter(ax,s):
 if hasattr(ax,'text2D'):ax.text2D(.01,.98,s,transform=ax.transAxes,fontsize=11,fontweight='bold',va='top')
 else:ax.annotate(s,(0,1),xycoords='axes fraction',xytext=(-24,8),textcoords='offset points',fontsize=11,fontweight='bold',annotation_clip=False)
def export(fig,name,axes,ids,rows=None,cols=None):
 fig.canvas.draw()
 require_matplotlib_panel_alignment(fig,axes=axes,panel_ids=ids,row_groups=rows,column_groups=cols,
  json_out=O/'qa'/f'{name}.alignment.json',overlay_svg=O/'qa'/f'{name}.alignment.svg',strict=True)
 fig.savefig(O/'preview'/f'{name}.png',dpi=300)
 if FINAL:
  fig.savefig(O/f'{name}.pdf',dpi=DPI)
  fig.savefig(O/f'{name}.svg',dpi=DPI);normalize(O/f'{name}.svg')
  fig.savefig(O/f'{name}.png',dpi=DPI)
 plt.close(fig)
def state(d,v='mapa'):
 return np.select([d[v].isna()|d[v+'dontdo'].isna(),d[v+'dontdo'].eq(1),d[v].eq(1)],[3,2,1],default=0)
def manifest():
 cp=O/'qa'/'numerical_checks.json';old_checks=json.loads(cp.read_text(encoding='utf-8')) if cp.exists() else []
 unique={json.dumps(x,sort_keys=True):x for x in old_checks+CHECKS};cp.write_text(json.dumps(list(unique.values()),indent=2),encoding='utf-8')
 pp=O/'provenance.json';old=json.loads(pp.read_text(encoding='utf-8')) if pp.exists() else {}
 inputs={x['path']:x for x in old.get('input_sha256',[])}
 def portable(p):
  if p.is_relative_to(RESULT):return 'result/'+p.relative_to(RESULT).as_posix()
  if p.is_relative_to(O):return 'generated/'+p.relative_to(O).as_posix()
  raise ValueError(f'Unmapped plotting source: {p}')
 inputs.update({portable(p):{'path':portable(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in INPUTS})
 pp.write_text(json.dumps({'input_sha256':list(inputs.values()),
  'scope':'Descriptive reorganization of observed data and saved outputs. No refitting, new predictions or hypothesis tests.',
  'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}},indent=2),encoding='utf-8')

def cohort(year):
 return read(RESULT/'cohorts'/f'{year}-{year+6}_final.csv')

def first_wave(d,year):
 a=read(RESULT/'cohorts'/f'{year}-{year+6}_audit.csv').set_index('hhidpn').reindex(d.hhidpn)
 cols=[f'dementia_{year+v}' for v in [2,4,6]];z=a[cols].eq(1).to_numpy()
 assert np.array_equal(z.any(axis=1).astype(int),d.y.to_numpy())
 return np.where(z.any(axis=1),2+2*z.argmax(axis=1),-1)

def alluvial(ax,seq):
 ss=sorted(seq.sequence); ns=dict(zip(seq.sequence,seq.n));n=sum(ns.values());m=len(ss[0]);codes=['N','D','?'];gap=.027
 pos={};bounds={}
 for t in range(m):
  base=0.
  for code in codes:
   q=[s for s in ss if s[t]==code];start=base
   for s in q:pos[s,t]=base;base+=ns[s]/n
   bounds[t,code]=(start,base);base+=gap
 for s in sorted(ss,key=lambda s:ns[s],reverse=True):
  color=WINE if 'D' in s else PURPLE if '?' not in s else GRAY
  for t in range(m-1):
   y0=pos[s,t];y1=pos[s,t+1];w=ns[s]/n
   verts=[(t+.025,y0),(t+.48,y0),(t+.52,y1),(t+.975,y1),
          (t+.975,y1+w),(t+.52,y1+w),(t+.48,y0+w),(t+.025,y0+w),(t+.025,y0)]
   codesp=[MPath.MOVETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4,MPath.LINETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4,MPath.CLOSEPOLY]
   ax.add_patch(PathPatch(MPath(verts,codesp),facecolor=color,alpha=.90 if 'D' in s else .82,edgecolor='none'))
 for (t,code),(a,b) in bounds.items():
  if b>a:ax.plot([t,t],[a,b],color=INK,lw=1.5)
  if t==m-1 and b>a:ax.text(m-1+.075,(a+b)/2,code,ha='left',va='center',fontsize=7)
 ax.set(xlim=(-.12,m-1+.05),ylim=(-.01,1+3*gap),xticks=range(m),xticklabels=[0,2,4,6],xlabel='Years from baseline',yticks=[])
 for sp in ax.spines.values():sp.set_visible(False)
 ax.invert_yaxis();assert n==seq.n.sum()

def domains(cols):
 groups={'Family':['ageym','gender','child','livpar','livsib','momage','dadage'],
 'Context':['edstg','lbrf','mstat','cendiv','raceeth'],
 'Health':['shlt','slfmem','height','weight','smokev','smoken','drink','vigact'],
 'Mood':['effort','depres','sleepr','whappy','flone','fsad','going','enlife'],
 'Disease':['hibpe','diabe','cancre','lunge','hearte','stroke','arthre'],
 'Care':['hosp','nrshom','nrstim','nrsnit','doctim','hlthlm','proxy']}
 out={};used=set()
 for g,vs in groups.items():
  cs=[c for c in cols if any(c==v or c.startswith(v) and v in ['edstg','lbrf','mstat','cendiv','raceeth'] for v in vs)]
  out[g]=cs;used.update(cs)
 out['Function']=[c for c in cols if c not in used]
 assert set(c for v in out.values() for c in v)==set(cols)
 return out

def kde2(points,reference,gridsize=96):
 """Full-covariance Gaussian KDE; all groups share the same pooled bandwidth matrix."""
 ref=np.asarray(reference,float);x=np.asarray(points,float)
 H=np.cov(ref.T)*len(ref)**(-1/3)
 L=np.linalg.cholesky(H);std=np.sqrt(np.diag(H))
 gx=np.linspace(ref[:,0].min()-4.5*std[0],ref[:,0].max()+4.5*std[0],gridsize)
 gy=np.linspace(ref[:,1].min()-4.5*std[1],ref[:,1].max()+4.5*std[1],gridsize)
 X,Y=np.meshgrid(gx,gy);grid=np.column_stack([X.ravel(),Y.ravel()])
 wx=np.linalg.solve(L,x.T).T; wg=np.linalg.solve(L,grid.T).T
 density=np.empty(len(grid));normalizer=2*np.pi*np.sqrt(np.linalg.det(H))
 for start in range(0,len(grid),256):
  dist=cdist(wg[start:start+256],wx,'sqeuclidean')
  density[start:start+256]=np.exp(-.5*dist).mean(axis=1)/normalizer
 Z=density.reshape(X.shape);area=(gx[1]-gx[0])*(gy[1]-gy[0]);mass=float(Z.sum()*area)
 order=np.argsort(Z.ravel())[::-1];cum=np.cumsum(Z.ravel()[order]*area)
 levels={m:float(Z.ravel()[order[min(np.searchsorted(cum,m),len(order)-1)]]) for m in [.5,.8]}
 check('KDE grid integrates to one',abs(mass-1)<.025,{'n':len(x),'mass':mass,'H':H.tolist()})
 return X,Y,Z,levels,H,mass

def kde1(values,grid,bandwidth):
 x=np.asarray(values,float)
 return norm.pdf((grid[:,None]-x[None,:])/bandwidth).mean(axis=1)/bandwidth

def bounded_rank_kde(points,gridsize=100):
 """Reflect Gaussian probability at the [0.5, 60.5] rank-domain boundaries."""
 x=np.asarray(points,float);H=np.cov(x.T)*len(x)**(-1/3);L=np.linalg.cholesky(H)
 lo,hi=.5,60.5;step=(hi-lo)/gridsize
 grid=np.linspace(lo+step/2,hi-step/2,gridsize);X,Y=np.meshgrid(grid,grid)
 wx=np.linalg.solve(L,x.T).T;Z=np.zeros(X.shape)
 for qx in [X,2*lo-X,2*hi-X]:
  for qy in [Y,2*lo-Y,2*hi-Y]:
   q=np.column_stack([qx.ravel(),qy.ravel()]);wq=np.linalg.solve(L,q.T).T
   density=np.exp(-.5*cdist(wq,wx,'sqeuclidean')).mean(axis=1)/(2*np.pi*np.sqrt(np.linalg.det(H)))
   Z+=density.reshape(X.shape)
 area=step**2;mass=float(Z.sum()*area);order=np.argsort(Z.ravel())[::-1];cum=np.cumsum(Z.ravel()[order]*area)
 levels={m:float(Z.ravel()[order[min(np.searchsorted(cum,m),len(order)-1)]]) for m in [.5,.8]}
 check('Reflected rank KDE integrates to one',abs(mass-1)<.01,{'n':len(x),'mass':mass,'H':H.tolist(),'domain':[lo,hi]})
 return X,Y,Z,levels,H,mass

def hdr(ax,K,color,bounded=False):
 X,Y,Z,lev,_,_=K
 if bounded:
  # Draw only outlines; include the rank-domain boundary wherever an HDR meets it.
  for mass,style,width in [(.8,'--',.9),(.5,'-',1.4)]:
   region=ax.contourf(X,Y,Z,levels=[lev[mass],Z.max()+1],colors='none')
   paths=region.get_paths();region.remove()
   for path in paths: ax.add_patch(PathPatch(path,fill=False,edgecolor=color,lw=width,ls=style))
  return
 ax.contour(X,Y,Z,levels=[lev[.8]],colors=[color],linewidths=.9,linestyles='--')
 ax.contour(X,Y,Z,levels=[lev[.5]],colors=[color],linewidths=1.4)
