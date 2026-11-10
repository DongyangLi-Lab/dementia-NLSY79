"""Five evidence stories; full observations, saved model outputs, no heat maps."""
import argparse,json,hashlib
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import common as C
from common import O,REPO,PREP,WINE,PURPLE,GOLD,INK,GRAY,BLUE,LIGHT,read,save,check,letter
from audit_panel_alignment import require_matplotlib_panel_alignment

from types import SimpleNamespace
args=SimpleNamespace(final=True);C.FINAL=True
plt.rcParams.update({'font.family':'Arial','pdf.fonttype':42,'svg.fonttype':'none'})
def export(fig,name,axes,ids,rows=None,cols=None):
 fig.canvas.draw()
 require_matplotlib_panel_alignment(fig,axes=axes,panel_ids=ids,row_groups=rows,column_groups=cols,
  json_out=O/'qa'/f'{name}.alignment.json',overlay_svg=O/'qa'/f'{name}.alignment.svg',strict=True)
 fig.savefig(O/'preview'/f'{name}.png',dpi=300)
 if args.final:
  fig.savefig(O/f'{name}.pdf',dpi=C.DPI)
  fig.savefig(O/f'{name}.svg',dpi=C.DPI);C.normalize(O/f'{name}.svg')
  fig.savefig(O/f'{name}.png',dpi=C.DPI)
 plt.close(fig)
def kde_cached(name,points,reference):
 path=O/'cache'/f'{name}.npz'
 digest=hashlib.sha256(np.asarray(points,float).tobytes()+np.asarray(reference,float).tobytes()).hexdigest()
 if path.exists():
  z=np.load(path)
  if str(z['digest'])==digest:
   K=(z['X'],z['Y'],z['Z'],{.5:float(z['hdr50']),.8:float(z['hdr80'])},z['H'],float(z['mass']))
   check('Cached KDE mass',abs(K[-1]-1)<.025,{'name':name,'mass':K[-1]});return K
 K=C.kde2(points,reference,gridsize=100)
 np.savez_compressed(path,X=K[0],Y=K[1],Z=K[2],hdr50=K[3][.5],hdr80=K[3][.8],H=K[4],mass=K[5],digest=digest)
 return K

def fig1():
 seq=read(PREP/'fig1_observation/observation_sequences.csv')
 fig=plt.figure(figsize=(190/25.4,210/25.4));axes=[]
 for i,year in enumerate([2000,2010]):
  a=fig.add_axes([.10+i*.49,.68,.37,.235]);s=seq[(seq.baseline==year)&(seq.horizon==6)]
  C.alluvial(a,s);a.set_title(str(year));letter(a,'AB'[i]);axes.append(a);save(s,f'Fig1_{year}_observation_paths')
  check('Observation path conservation',s.n.sum()==len(C.cohort(year)),{'baseline':year,'n':int(s.n.sum())})
 for i,year in enumerate([2000,2010]):
  a=fig.add_axes([.10+i*.49,.365,.37,.215]);d=C.cohort(year);fw=C.first_wave(d,year)
  edges=np.arange(50,104,2);ss=[]
  for k,col in zip([2,4,6],[PURPLE,WINE,GOLD]):
   q=d.loc[fw==k,'ageym'];ss.append(q.to_numpy())
  a.hist(ss,bins=edges,stacked=True,color=[PURPLE,WINE,GOLD],edgecolor='white',linewidth=.25,label=['2 years','4 years','6 years'])
  a.set(xlabel='Baseline age (years)',ylabel='First observed cases',xlim=(49,102));letter(a,'CD'[i]);axes.append(a)
  source=pd.DataFrame({'baseline':year,'age':d.ageym,'first_observed_year':fw,'outcome':d.y})
  save(source,f'Fig1_{year}_first_observation_age');check('Case histogram conservation',sum(map(len,ss))==int(d.y.sum()))
  check('All case ages inside histogram bins',sum(np.histogram(v,bins=edges)[0].sum() for v in ss)==int(d.y.sum()))
  if i==0:a.legend(fontsize=6.5,loc='upper right',handlelength=1)
 e=fig.add_axes([.10,.085,.37,.18]);f=fig.add_axes([.59,.085,.37,.18]);axes.extend([e,f]);out=[]
 for year,col in [(2000,PURPLE),(2010,WINE)]:
  s=seq[(seq.baseline==year)&(seq.horizon==6)].sort_values('n',ascending=False)
  e.scatter(np.arange(1,len(s)+1),s.n,s=12,color=col,alpha=.85,label=str(year))
  d=C.cohort(year);q=d.groupby('ageym').y.agg(n='size',cases='sum').sort_index(ascending=False)
  xx=np.r_[0,q.n.cumsum()/len(d)];yy=np.r_[0,q.cases.cumsum()/d.y.sum()]
  f.fill_between(xx,xx,yy,step='post',color=col,alpha=.12);f.step(xx,yy,where='post',color=col,lw=1.5,label=str(year))
  out.extend({'baseline':year,'age_threshold':age,'participant_share':x,'case_share':y} for age,x,y in zip(q.index,xx[1:],yy[1:]))
  save(s,f'Fig1_{year}_path_frequency_rank')
 e.set(xscale='log',yscale='log',xlabel='Observation-pattern rank',ylabel='Participants');e.legend(fontsize=7)
 e.set_xticks([1,10,30],['1','10','30']);e.set_yticks([1,10,100,1000,10000],['1','10','100','1,000','10,000']);e.minorticks_off()
 f.plot([0,1],[0,1],color=GRAY,lw=.7,ls=':');f.set(xlim=(0,1),ylim=(0,1),xlabel='Participants, oldest first',ylabel='Observed-case share')
 letter(e,'E');letter(f,'F');save(pd.DataFrame(out),'Fig1_age_case_concentration')
 export(fig,'Fig1_observation_window',axes,list('ABCDEF'),[['A','B'],['C','D'],['E','F']])

def fig2():
 d=C.cohort(2010);pred=[c for c in d if c not in ['hhidpn','demcls','y']];complete=d[pred].notna().all(axis=1)
 d['coverage']=np.select([d.proxy.eq(1),complete],['Proxy','Complete self'],default='Incomplete self')
 fields=['ageym','height','weight'];labels=['Age (years)','Height (m)','Weight (kg)'];groups=['Complete self','Incomplete self','Proxy'];colors=[BLUE,'#C77710','#7A3FA1']
 valid=d[fields].notna().all(axis=1);v=d.loc[valid].copy();pool=v[fields].to_numpy()
 save(d[fields+['coverage','y']].assign(in_joint_matrix=valid),'Fig2_joint_observations')
 meta=[]
 for group in groups:meta.append({'group':group,'n_total':int(d.coverage.eq(group).sum()),'n_joint':int(v.coverage.eq(group).sum()),'n_not_joint':int((d.coverage.eq(group)&~valid).sum())})
 save(pd.DataFrame(meta),'Fig2_joint_coverage_counts');check('Exclusive coverage groups',sum(x['n_total'] for x in meta)==len(d))
 print('Fig2 joint density calculation',flush=True)
 kernels={}
 for r,c in combinations(range(3),2):
  ref=v[[fields[c],fields[r]]].to_numpy()
  for g in groups:
   pp=v.loc[v.coverage.eq(g),[fields[c],fields[r]]].to_numpy()
   kernels[r,c,g]=kde_cached(f'F2_{r}_{c}_{g.replace(" ","_")}',pp,ref)
   k=kernels[r,c,g];save(pd.DataFrame({'x':k[0].ravel(),'y':k[1].ravel(),'density':k[2].ravel()}),f'Fig2_density_{r}_{c}_{g.replace(" ","_")}')
 fig=plt.figure(figsize=(190/25.4,215/25.4));axes=[];ids=[];matrix=[]
 limits=[]
 for f in fields:
  x=v[f].to_numpy();bw=x.std(ddof=1)*len(x)**(-.2);limits.append((x.min()-4.5*bw,x.max()+4.5*bw))
 for r in range(3):
  row=[]
  for c in range(3):
   ax=fig.add_axes([.10+c*.215,.755-r*.185,.175,.145]);ident=f'M{r}{c}';axes.append(ax);ids.append(ident);row.append(ident)
   if r==c:
    grid=np.linspace(*limits[c],260);bw=v[fields[c]].std(ddof=1)*len(v)**(-.2)
    for g,col in zip(groups,colors):
     den=C.kde1(v.loc[v.coverage.eq(g),fields[c]],grid,bw)
     ax.fill_between(grid,den,color=col,alpha=.16);ax.plot(grid,den,color=col,lw=1.1)
    ax.set(xlim=limits[c],yticks=[])
   elif r<c:
    for g,col in zip(groups,colors):C.hdr(ax,kernels[r,c,g],col)
    ax.set(xlim=limits[c],ylim=limits[r])
   else:
    for g,col in zip(groups,colors):
     q=v[v.coverage.eq(g)];ax.scatter(q[fields[c]],q[fields[r]],s=1.5,c=col,alpha=.23,edgecolor='none',rasterized=True)
    ax.set(xlim=limits[c],ylim=limits[r])
   if r==2:ax.set_xlabel(labels[c],labelpad=3)
   else:ax.set_xticklabels([])
   if c==0:ax.set_ylabel('Density' if r==0 else labels[r],labelpad=3)
   else:ax.set_yticklabels([])
   ax.tick_params(length=2,pad=2)
  matrix.append(row)
 letter(axes[0],'A')
 fig.legend(handles=[Patch(facecolor=c,label=g) for g,c in zip(groups,colors)],loc='center',bbox_to_anchor=(.43,.96),ncol=3,fontsize=7,handlelength=1,columnspacing=1.0)
 fig.legend(handles=[Line2D([],[],color=INK,lw=1.2,label='50% mass'),Line2D([],[],color=INK,lw=.9,ls='--',label='80% mass')],loc='center',bbox_to_anchor=(.43,.925),ncol=2,fontsize=6.5,handlelength=1.4)
 # Three dense distribution/mass views probe missingness rather than repeating the joint matrix.
 b=fig.add_axes([.79,.755,.175,.145]);cc=fig.add_axes([.79,.570,.175,.145]);ee=fig.add_axes([.79,.385,.175,.145])
 missing=d[pred].isna().sum(axis=1)
 adl_masses=[]
 for gi,(g,col) in enumerate(zip(groups,colors)):
  q=np.sort(missing[d.coverage.eq(g)]);b.step(q,np.arange(1,len(q)+1)/len(q),where='post',color=col,lw=1.2)
  z=d.loc[d.coverage.eq(g),'doctim'].dropna().sort_values().to_numpy();cc.step(np.log1p(z),np.arange(1,len(z)+1)/len(z),where='post',color=col,lw=1.2)
  group=d[d.coverage.eq(g)];ad=group.adl5a;case=group.y.eq(1);no=[];yes=[]
  for score in range(7):
   hit=ad.eq(score) if score<6 else ad.isna();n0=int((hit&~case).sum());n1=int((hit&case).sum());no.append(n0/len(ad));yes.append(n1/len(ad))
   adl_masses.append({'group':g,'ADL_count':score if score<6 else 'Unrecorded','n_noncase':n0,'n_case':n1,'group_n':len(ad)})
  ee.bar(np.arange(7)+(gi-1)*.24,no,width=.23,facecolor='white',edgecolor=col,linewidth=.45)
  ee.bar(np.arange(7)+(gi-1)*.24,yes,bottom=no,width=.23,color=col,edgecolor=col,linewidth=.25)
 b.set(xlabel='Missing inputs',ylabel='Cumulative fraction',ylim=(0,1.03));cc.set(xlabel='ln(1 + visits)',ylabel='Cumulative fraction',ylim=(0,1.03))
 ee.set(xlabel='ADL difficulties',ylabel='Group fraction',xticks=[0,3,6],xticklabels=['0','3','NA'],xlim=(-.5,6.5))
 for a,l in [(b,'B'),(cc,'C'),(ee,'D')]:letter(a,l);axes.append(a);ids.append(l)
 save(d[['coverage','ageym','doctim','adl5a','y']].assign(missing_inputs=missing),'Fig2_coverage_marginals')
 save(pd.DataFrame(adl_masses),'Fig2_ADL_outcome_mass');check('ADL outcome mass conservation',sum(x['n_case']+x['n_noncase'] for x in adl_masses)==len(d))
 f=fig.add_axes([.10,.085,.865,.19]);m=read(PREP/'fig2_coverage/missingness_by_predictor.csv');order=m.groupby('variable').fraction_missing.max().sort_values(ascending=False).index.tolist()
 for base,h,col,mark in [(2000,6,PURPLE,'o'),(2010,6,WINE,'D'),(2000,8,BLUE,'s'),(2010,8,GOLD,'^')]:
  q=m[(m.baseline==base)&(m.horizon==h)].set_index('variable').loc[order]
  f.plot(range(1,61),100*q.fraction_missing,color=col,lw=.6,alpha=.65)
  f.scatter(range(1,61),100*q.fraction_missing,color=col,marker=mark,s=9,label=f'{base}, {h}y',zorder=3)
 f.set(xlim=(.2,60.8),xticks=[1,10,20,30,40,50,60],xlabel='Predictors ordered by maximum missingness (all 60)',ylabel='Unrecorded (%)');letter(f,'E');f.legend(fontsize=6.5,ncol=4,loc='upper right',handlelength=.9,columnspacing=1)
 axes.append(f);ids.append('E');save(m.assign(display_rank=m.variable.map({s:i+1 for i,s in enumerate(order)})),'Fig2_full_missingness_spectrum')
 export(fig,'Fig2_coverage_joint_matrix',axes,ids,matrix,[[f'M{r}{c}' for r in range(3)] for c in range(3)]+[['B','C','D']])

def fig3():
 d=C.cohort(2010);d['navigation']=C.state(d);valid=d.ageym.notna()&d.weight.notna()&d.height.notna()&d.height.gt(0)
 q=d.loc[valid].copy();q['BMI']=q.weight/q.height.pow(2);ref=q[['ageym','BMI']].to_numpy();K=[];meta=[]
 for i in range(3):
  s=q[q.navigation.eq(i)];k=kde_cached(f'F3_map{i}_age_BMI',s[['ageym','BMI']].to_numpy(),ref);K.append(k)
  meta.append({'navigation':C.NAV[i],'n_total':int(d.navigation.eq(i).sum()),'n_density':len(s),'mass':k[-1],'hdr50':k[3][.5],'hdr80':k[3][.8],'H':json.dumps(k[4].tolist())})
  save(pd.DataFrame({'age':k[0].ravel(),'BMI':k[1].ravel(),'density':k[2].ravel()}),f'Fig3_joint_density_map{i}')
 save(pd.DataFrame(meta),'Fig3_density_metadata')
 save(d[['ageym','height','weight','navigation','y']].assign(BMI=d.weight/d.height.pow(2),eligible_for_shared_reference=valid,in_displayed_density=valid&d.navigation.lt(3)),'Fig3_density_observations')
 save(pd.DataFrame([{'navigation':C.NAV[i],'n_total':int(d.navigation.eq(i).sum()),'n_body_pair':int((d.navigation.eq(i)&valid).sum()),'n_body_pair_unavailable':int((d.navigation.eq(i)&~valid).sum()),'displayed_as_density':i<3} for i in range(4)]),'Fig3_density_inclusion_counts')
 fig=plt.figure(figsize=(190/25.4,106/25.4));axes=[];ids=[];zmax=max(k[2].max() for k in K)*1.04
 for i,(k,col) in enumerate(zip(K,C.NC[:3])):
  x=.06+i*.317
  a=fig.add_axes([x-.018,.50,.295,.43],projection='3d');X,Y,Z,lev,H,mass=k
  cm=LinearSegmentedColormap.from_list('density',['#FAF8FA',col])
  a.plot_surface(X,Y,Z,cmap=cm,vmin=0,vmax=zmax,linewidth=.16,edgecolor='#746B78',rstride=3,cstride=3,antialiased=True,rasterized=True,alpha=.97)
  a.contour(X,Y,Z,zdir='z',offset=0,levels=[lev[.8],lev[.5]],colors=[col],linewidths=[.7,1.1])
  a.set(xlim=(X.min(),X.max()),ylim=(Y.min(),Y.max()),zlim=(0,zmax),xlabel='Age',ylabel='BMI',zlabel='Density' if i==0 else '');a.view_init(27,-58)
  a.set_box_aspect((1.15,1,.60));a.tick_params(labelsize=5.8,pad=0);a.xaxis.labelpad=-1;a.yaxis.labelpad=-1;a.zaxis.labelpad=0
  a.set_xticks([50,75,100]);a.set_yticks([20,40,60]);a.locator_params(axis='z',nbins=3)
  a.set_title(C.NAV[i],fontsize=7.5,pad=1);letter(a,'ABC'[i]);axes.append(a);ids.append('ABC'[i])
  b=fig.add_axes([x,.10,.255,.29]);s=q[q.navigation.eq(i)]
  for out,marker in [(0,'o'),(1,'D')]:
   v=s[s.y.eq(out)];b.scatter(v.ageym,v.BMI,s=1.6 if out==0 else 4.2,marker=marker,facecolor='none' if out==0 else col,edgecolor=col,linewidth=.17 if out==0 else 0,alpha=.18 if out==0 else .75,rasterized=True)
  C.hdr(b,k,col);b.set(xlim=(X.min(),X.max()),ylim=(Y.min(),Y.max()),xlabel='Age (years)',ylabel='BMI (kg/m²)');b.tick_params(labelsize=6.5)
  letter(b,'DEF'[i]);axes.append(b);ids.append('DEF'[i])
 export(fig,'Fig3_density_component',axes,ids,[list('ABC'),list('DEF')])
 if args.final:
  import pymupdf
  out=pymupdf.open();page=out.new_page(width=190*72/25.4,height=226*72/25.4)
  a=pymupdf.open(O/'Fig3_density_component.pdf');bpath=O/'selected_component'/'education_navigation_violin_box_overlay.pdf';C.INPUTS.add(bpath)
  b=pymupdf.open(bpath)
  page.show_pdf_page(pymupdf.Rect(0,0,190*72/25.4,106*72/25.4),a,0)
  page.show_pdf_page(pymupdf.Rect(0,114*72/25.4,190*72/25.4,226*72/25.4),b,0)
  page.insert_text((4*72/25.4,115.5*72/25.4),'G',fontsize=11,fontname='hebo')
  out.save(O/'Fig3_functional_landscape.pdf',deflate=True)
  pix=page.get_pixmap(matrix=pymupdf.Matrix(C.DPI/72,C.DPI/72),alpha=False);pix.set_dpi(C.DPI,C.DPI);pix.save(O/'Fig3_functional_landscape.png')
  pix=page.get_pixmap(matrix=pymupdf.Matrix(300/72,300/72),alpha=False);pix.set_dpi(300,300);pix.save(O/'preview'/'Fig3_functional_landscape.png')
  from svg_composite import compose
  compose(O/'Fig3_density_component.svg',O/'selected_component'/'education_navigation_violin_box_overlay.svg',O/'Fig3_functional_landscape.svg')
  (O/'qa'/'Fig3_composition.json').write_text(json.dumps({'component_alignment':'Fig3_density_component.alignment.json','lower_component':'Selected v3 layout, regenerated from revised final survivors at 190 x 112 mm.','exemption':'Lower full-width panel G is intentionally incomparable with the six upper density panels. No equal-width claim across these different spans.','n_violins_in_suite':'One designated figure: Fig3 panel G only.'},indent=2),encoding='utf-8')

def fig4():
 p=read(PREP/'fig4_signal/importance_permutation_h6.csv');sh=read(PREP/'fig4_signal/importance_shap_h6.csv');meta=read(PREP/'fig4_signal/prediction_reconstruction_check.csv')
 models=['LR','GNB','DT','LightGBM','CatBoost','RealMLP','GAT','TabICL'];p=p[p.model.isin(models)].copy()
 ranks=p.pivot(index='variable',columns='model',values='rank').loc[:,models];order=ranks.median(axis=1).sort_values().index;ranks=ranks.loc[order]
 paired=p.merge(sh,on=['horizon','model','variable'],suffixes=('_perm','_shap'));cores={'ageym':PURPLE,'edstg':WINE,'mapa':GOLD}
 summary=pd.DataFrame({'variable':ranks.index,'median_rank':ranks.median(axis=1),'rank_IQR':ranks.quantile(.75,axis=1)-ranks.quantile(.25,axis=1),'n_top10':ranks.le(10).sum(axis=1)})
 save(summary.reset_index(drop=True),'Fig4_consensus_and_dispersion');save(p,'Fig4_all_480_importance_estimates');save(paired,'Fig4_paired_explanation_ranks');save(meta,'Fig4_prediction_reconstruction')
 fig=plt.figure(figsize=(190/25.4,208/25.4));a=fig.add_axes([.10,.61,.49,.29]);b=fig.add_axes([.72,.61,.25,.29])
 for v in ranks.index:a.plot(range(8),ranks.loc[v],color=cores.get(v,GRAY),alpha=.95 if v in cores else .3,lw=1.7 if v in cores else .55,zorder=4 if v in cores else 1)
 for i in range(8):a.axvline(i,color=LIGHT,lw=.5,zorder=0)
 a.set(ylim=(62,0),yticks=[1,10,20,40,60],xlim=(-.1,7.1),xticks=range(8),ylabel='Permutation rank')
 a.set_xticklabels(models,rotation=45,ha='right',rotation_mode='anchor');letter(a,'A')
 for _,r in summary.iterrows():
  b.scatter(r.median_rank,r.rank_IQR,s=12+12*r.n_top10,color=cores.get(r.variable,GRAY),edgecolor='white',linewidth=.25,alpha=.85)
 b.set(xlim=(-3,63),ylim=(-3,max(summary.rank_IQR)+3),xticks=[0,20,40,60],yticks=[0,10,20,30],xlabel='Median rank',ylabel='Rank IQR');letter(b,'B')
 b.legend(handles=[b.scatter([],[],s=12+12*n,color=GRAY,edgecolor='white',linewidth=.25,label=str(n)) for n in [0,4,8]],title='Models in top 10',loc='lower center',bbox_to_anchor=(.5,1.04),ncol=3,fontsize=6.5,title_fontsize=6.5,handlelength=1,columnspacing=.7,handletextpad=.35)
 a.legend(handles=[Line2D([],[],color=c,lw=1.7,label=l) for l,c in [('Age',PURPLE),('Education',WINE),('Map use',GOLD)]],loc='upper left',bbox_to_anchor=(0,-.23),ncol=3,fontsize=7,handlelength=1.1,columnspacing=1)
 c=fig.add_axes([.10,.12,.37,.31]);dax=fig.add_axes([.60,.12,.37,.31]);K=C.bounded_rank_kde(paired[['rank_perm','rank_shap']].to_numpy())
 for v,col in [('other',GRAY),*cores.items()]:
  q=paired[~paired.variable.isin(cores)] if v=='other' else paired[paired.variable.eq(v)]
  c.scatter(q.rank_perm,q.rank_shap,s=9 if v=='other' else 19,c=col,alpha=.55 if v=='other' else .95,edgecolor='white',linewidth=.2,zorder=3)
 C.hdr(c,K,PURPLE,bounded=True);c.plot([1,60],[1,60],color=INK,lw=.8,ls=':');c.set(xlim=(0,61),ylim=(0,61),xlabel='Permutation rank',ylabel='SHAP rank');letter(c,'C')
 save(pd.DataFrame({'permutation_rank':K[0].ravel(),'shap_rank':K[1].ravel(),'density':K[2].ravel()}),'Fig4_rank_joint_density')
 other=p[~p.variable.isin(cores)]
 dax.scatter(other['rank'],other.importance_AUPRC_drop,c=GRAY,s=4,alpha=.65,edgecolor='none')
 for v,col in cores.items():
  q=p[p.variable.eq(v)];dax.scatter(q['rank'],q.importance_AUPRC_drop,color=col,s=9,zorder=4)
 dax.set_yscale('symlog',linthresh=.003,linscale=.7);dax.set(xlim=(0,61),xlabel='Permutation rank',ylabel='Importance (AUPRC drop)')
 dax.set_yticks([-.005,0,.005,.05,.25],['−.005','0','.005','.05','.25']);dax.axhline(0,color=INK,lw=.7,ls=':');letter(dax,'D')
 check('Explanation units',len(p)==480 and len(paired)==480 and ranks.shape==(60,8))
 export(fig,'Fig4_signal_consensus',[a,b,c,dax],list('ABCD'),[['A','B'],['C','D']])

def fig5():
 models=['LR','CatBoost','TabICL'];mc=dict(zip(models,[INK,GOLD,WINE]))
 cal=read(PREP/'fig5_reliability/calibration_curves.csv');cal=cal[cal.population.eq('all')&cal.model.isin(models)].copy()
 dc=read(PREP/'fig5_reliability/decision_curves.csv');dc=dc[dc.population.eq('all')&dc.model.isin(models)].copy()
 cal['residual']=cal.observed-cal.predicted;cal['within_curve_bin_share']=cal.n/cal.groupby(['horizon','model','setting']).n.transform('sum')
 save(cal,'Fig5_calibration_bin_residuals');fig=plt.figure(figsize=(190/25.4,215/25.4));axes=[]
 yl=(min(-.01,cal.residual.min()-.005),max(.01,cal.residual.max()+.005))
 for i,h in enumerate([6,8]):
  a=fig.add_axes([.10+i*.49,.70,.37,.205]);a.axhline(0,color=GRAY,lw=.8,ls=':')
  for m in models:
   for setting,mark in [('temporal','o'),('internal (pooled repeats)','s')]:
    q=cal[cal.horizon.eq(h)&cal.model.eq(m)&cal.setting.eq(setting)]
    a.scatter(q.predicted,q.residual,s=12+170*q.within_curve_bin_share,marker=mark,facecolor=mc[m] if setting=='temporal' else 'none',edgecolor=mc[m],linewidth=.7,alpha=.88)
  a.set(xlim=(-.015,.72),ylim=yl,xlabel='Mean predicted risk',ylabel='Observed − predicted');a.set_title(f'{h}-year');letter(a,'AB'[i]);axes.append(a)
 paired=read(O/'source_data/decision_difference_ci.csv')
 paired=paired[paired.model.eq('TabICL') & paired.reference.isin(['LR','CatBoost'])].copy()
 assert len(paired)==316
 limits=(1000*paired.lo.min()-.35,1000*paired.hi.max()+.35)
 comp_colors={'LR':'#174A8B','CatBoost':'#B53A24'}
 for col,h in enumerate([6,8]):
  a=fig.add_axes([.10+.49*col,.39,.37,.20]);axes.append(a)
  for ref in ['LR','CatBoost']:
   q=paired[paired.horizon.eq(h)&paired.reference.eq(ref)].sort_values('threshold')
   assert len(q)==79 and np.all(np.diff(q.threshold)>0)
   a.fill_between(q.threshold.to_numpy(),1000*q.lo.to_numpy(),1000*q.hi.to_numpy(),color=comp_colors[ref],alpha=.24,lw=0,zorder=1)
   a.plot(q.threshold,1000*q.difference,color=comp_colors[ref],lw=1.5,label=f'TabICL − {ref}',zorder=3)
  a.axhline(0,color=GRAY,lw=.7,zorder=2)
  a.set(xlim=(.01,.4),ylim=limits,xlabel='Risk threshold',ylabel='Δ net benefit / 1,000')
  a.set_title(f'{h}-year paired comparison',loc='left',pad=7);letter(a,'CD'[col])
  if col==0:a.legend(fontsize=6.5,loc='lower left')
 save(paired,'Fig5_CD_paired_intervals')
 e=fig.add_axes([.10,.09,.37,.19]);f=fig.add_axes([.59,.09,.37,.19]);runs=[];thresholds=[]
 for h in [6,8]:
  for m in models:
   q=read(C.RESULT/f'figure_inputs/selection/final/h{h}/{m}/validation_grid.csv');q['model']=m;q['horizon']=h;runs.append(q)
   path=C.RESULT/f'figure_inputs/selection/final/h{h}/{m}/selection.json';C.INPUTS.add(path);sel=json.loads(path.read_text())
   thresholds.extend({'model':m,'horizon':h,'repeat':r['repeat'],'threshold':r['threshold']} for r in sel['per_repeat'])
 runs=pd.concat(runs,ignore_index=True);th=pd.DataFrame(thresholds)
 threshold_display=[]
 for j,m in enumerate(models):
  z=np.sort(runs.loc[runs.model.eq(m),'fit_seconds']);check('Positive runtime',np.all(z>0));e.step(z,np.arange(1,len(z)+1)/len(z),where='post',color=mc[m],lw=1.5)
  for h,offset,marker in [(6,-.14,'^'),(8,.14,'v')]:
   q=th[th.model.eq(m)&th.horizon.eq(h)].sort_values('repeat');xx=j+offset+np.linspace(-.065,.065,len(q))
   f.scatter(xx,q.threshold,s=14,marker=marker,facecolor=mc[m] if h==6 else 'none',edgecolor=mc[m],linewidth=.65,zorder=3)
   med=q.threshold.median();f.plot([j+offset-.09,j+offset+.09],[med,med],color=mc[m],lw=1.2,zorder=4)
   threshold_display.extend({'model':m,'horizon':h,'repeat':int(r),'threshold':float(t),'x_display':float(x)} for r,t,x in zip(q.repeat,q.threshold,xx))
 e.set(xscale='log',xlabel='Validation-fit time (s)',ylabel='Cumulative fraction',ylim=(0,1.03));e.set_xticks([.01,1,100],['0.01','1','100']);e.minorticks_off()
 f.set(xticks=range(3),xticklabels=models,xlim=(-.5,2.5),ylabel='Selected validation threshold');letter(e,'E');letter(f,'F');axes.extend([e,f])
 f.legend(handles=[Line2D([],[],marker='^',mfc=INK,mec=INK,ls='none',ms=3,label='6-year'),Line2D([],[],marker='v',mfc='none',mec=INK,ls='none',ms=3,label='8-year')],loc='lower center',bbox_to_anchor=(.5,1.035),fontsize=6.5,ncol=2,handlelength=.8,columnspacing=.8)
 save(pd.DataFrame(threshold_display),'Fig5_threshold_display_coordinates')
 save(runs,'Fig5_all_validation_search_runs');save(th,'Fig5_selected_thresholds')
 handles=[Line2D([],[],color=mc[m],lw=1.5,label=m) for m in models]+[Line2D([],[],marker='s',mfc='none',mec=INK,ls='none',label='Internal'),Line2D([],[],marker='o',mfc=INK,mec=INK,ls='none',label='Temporal')]
 fig.legend(handles=handles,loc='center',bbox_to_anchor=(.53,.025),ncol=5,fontsize=7,handlelength=1,columnspacing=1)
 export(fig,'Fig5_probability_reliability',axes,list('ABCDEF'),[['A','B'],['C','D'],['E','F']])
