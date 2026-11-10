"""One designated violin figure: pooled density, grouped boxes and all observed records."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from scipy.stats import gaussian_kde
from scipy.integrate import trapezoid
from audit_panel_alignment import require_matplotlib_panel_alignment
from svg_compat import normalize




def render(source, output, dpi=300):
 SOURCE=Path(source);O=Path(output);O.mkdir(parents=True,exist_ok=True)
 for p in ['preview','source_data','qa']:(O/p).mkdir(exist_ok=True)
 WINE='#A3133F';PURPLE='#553574';GOLD='#AD6500';INK='#25252B';GRAY='#88858B'
 EDU=['0–7','8–11','12','13+'];NAV=['No difficulty','Difficulty','Does not do','Unrecorded']
 NC=dict(zip(NAV,[PURPLE,WINE,GOLD,GRAY]))
 plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':8.5,'xtick.labelsize':8,
  'ytick.labelsize':8,'axes.linewidth':.95,'xtick.major.width':.85,'ytick.major.width':.85,
  'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none',
  'legend.frameon':False,'figure.facecolor':'white','savefig.facecolor':'white'})
 d=pd.read_csv(SOURCE);assert d.hhidpn.is_unique and d.y.isin([0,1]).all()
 dummy=d[['edstg2 8to11','edstg3 12','edstg4 13above']]
 assert dummy.fillna(0).sum(axis=1).le(1).all()
 d['education']=np.select([dummy.iloc[:,i].eq(1) for i in range(3)],EDU[1:],default=EDU[0])
 d.loc[dummy.isna().any(axis=1),'education']='Unrecorded'
 d['navigation']=np.select([d.mapa.isna()|d.mapadontdo.isna(),d.mapadontdo.eq(1),d.mapa.eq(1)],['Unrecorded','Does not do','Difficulty'],default='No difficulty')
 known=d.education.isin(EDU)&d.ageym.notna()
 plot=d.loc[known,['education','navigation','ageym','y']].copy()
 plot.rename(columns={'ageym':'age_years','y':'observed_dementia_6y'},inplace=True)
 plot.insert(0,'display_record',np.arange(1,len(plot)+1))
 plot.to_csv(O/'source_data'/'all_displayed_records.csv',index=False,encoding='utf-8-sig')
 omitted=d.loc[~known,['education','navigation','ageym','y']].copy()
 omitted.to_csv(O/'source_data'/'records_without_plot_education_or_age.csv',index=False,encoding='utf-8-sig')
 counts=plot.groupby(['education','navigation','observed_dementia_6y'],observed=True).age_years.agg(n='size',mean='mean',median='median',minimum='min',maximum='max').reset_index()
 counts.to_csv(O/'source_data'/'subgroup_counts_and_centres.csv',index=False,encoding='utf-8-sig')

 # All four densities share the same absolute bandwidth and maximum-width mapping.
 ages=plot.age_years.to_numpy();bandwidth=ages.std(ddof=1)*len(ages)**(-.2)
 grid=np.linspace(44,106,621)
 densities={}
 for e in EDU:
  a=plot.loc[plot.education.eq(e),'age_years'].to_numpy()
  kde=gaussian_kde(a,bw_method=bandwidth/a.std(ddof=1));pdf=kde(grid);mass=float(trapezoid(pdf,grid))
  assert abs(mass-1)<.005,(e,mass)
  densities[e]=(a,pdf,mass)
 peak=max(pdf.max() for _,pdf,_ in densities.values())
 pd.DataFrame([{'education':e,'age_years':g,'density':f,'common_bandwidth_years':bandwidth} for e,(_,pdf,_) in densities.items() for g,f in zip(grid,pdf)]).to_csv(O/'source_data'/'pooled_density_curves.csv',index=False,encoding='utf-8-sig')

 fig=plt.figure(figsize=(190/25.4,112/25.4));ax=fig.add_axes([.085,.14,.895,.68])
 rng=np.random.default_rng(371)  # Display-only jitter. No simulated measurements or downsampling.
 summary=[];point_rows=[]
 for j,e in enumerate(EDU):
  centre=j*2.1;group=plot[plot.education.eq(e)];a,pdf,mass=densities[e]
  # The neutral pooled distribution remains visually separate from subgroup boxes.
  anchor=centre-.09
  ax.fill_betweenx(grid,anchor-.62*pdf/peak,anchor,facecolor='#D4D0D5',edgecolor='#625D66',lw=.65,alpha=.95,zorder=1)
  # Three navigation categories, each split into observed non-case and case.
  positions=[];values=[];faces=[];edges=[];labels=[]
  for nav_i,nav in enumerate(NAV[:3]):
   for outcome in (0,1):
    v=group.loc[group.navigation.eq(nav)&group.observed_dementia_6y.eq(outcome),'age_years'].to_numpy()
    assert len(v)>0,(e,nav,outcome,len(v))
    positions.append(centre+.035+nav_i*.205+outcome*.087)
    values.append(v);faces.append('white' if outcome==0 else NC[nav]);edges.append(NC[nav]);labels.append((nav,outcome))
    q=np.quantile(v,[.05,.25,.5,.75,.95])
    summary.append({'education':e,'navigation':nav,'observed_dementia_6y':outcome,'n':len(v),
     'p05':q[0],'q25':q[1],'median':q[2],'q75':q[3],'p95':q[4],'mean':v.mean()})
  bp=ax.boxplot(values,positions=positions,widths=.072,whis=(5,95),showfliers=False,patch_artist=True,manage_ticks=False,
   showmeans=True,meanprops={'marker':'^','markerfacecolor':'white','markeredgecolor':INK,'markeredgewidth':.45,'markersize':2.9},
   medianprops={'color':INK,'linewidth':.9},whiskerprops={'color':'#625D66','linewidth':.65},capprops={'color':'#625D66','linewidth':.65})
  for box,face,edge in zip(bp['boxes'],faces,edges):box.set(facecolor=face,edgecolor=edge,linewidth=.75,alpha=.95,zorder=4)
  for k in ['whiskers','caps','medians','means']:
   for artist in bp[k]:artist.set_zorder(5)
  # Every participant with known education is plotted once. Unknown navigation stays visible in gray.
  for nav in NAV:
   for outcome in (0,1):
    s=group[group.navigation.eq(nav)&group.observed_dementia_6y.eq(outcome)]
    x=centre+.73+rng.uniform(0,.50,len(s))
    if outcome==0:
     ax.scatter(x,s.age_years,s=1.9,facecolor='none',edgecolor=NC[nav],alpha=.39,linewidths=.25,rasterized=True,zorder=2)
    else:
     ax.scatter(x,s.age_years,s=4.1,marker='D',facecolor=NC[nav],edgecolor='none',alpha=.77,rasterized=True,zorder=3)
    point_rows.extend([{'display_record':int(i),'x_display_only':float(xx)} for i,xx in zip(s.display_record,x)])
 ax.set(xlim=(-.82,7.70),ylim=(44,106),xticks=np.arange(4)*2.1+.265,xticklabels=EDU,
  yticks=[50,60,70,80,90,100],xlabel='Education (years)',ylabel='Baseline age (years)')
 ax.grid(axis='y',color='#DEDBDF',linewidth=.45,zorder=0);ax.set_axisbelow(True)
 ax.tick_params(axis='x',length=0,pad=7)

 # Two compact encoding keys; prose and all subgroup counts stay outside the figure.
 fig.legend(handles=[Patch(facecolor=NC[k],edgecolor=NC[k],label=k) for k in NAV],
  loc='center',bbox_to_anchor=(.52,.955),ncol=4,fontsize=7.5,handlelength=1.1,columnspacing=1.8,handletextpad=.45)
 no=(Patch(facecolor='white',edgecolor=INK,lw=.7),Line2D([],[],marker='o',mfc='none',mec=INK,ms=3,ls='none'))
 yes=(Patch(facecolor='#777278',edgecolor=INK,lw=.7),Line2D([],[],marker='D',mfc=INK,mec=INK,ms=3,ls='none'))
 fig.legend([no,yes],['No observed dementia','Observed dementia'],handler_map={tuple:HandlerTuple(ndivide=None,pad=.3)},
  loc='center',bbox_to_anchor=(.52,.895),ncol=2,fontsize=7.5,handlelength=2.0,columnspacing=2.3,handletextpad=.6)

 pd.DataFrame(summary).to_csv(O/'source_data'/'boxplot_quantiles_and_means.csv',index=False,encoding='utf-8-sig')
 pd.DataFrame(point_rows).to_csv(O/'source_data'/'display_jitter_coordinates.csv',index=False,encoding='utf-8-sig')
 assert len(point_rows)==len(plot) and len(set(p['display_record'] for p in point_rows))==len(plot)
 assert sum(len(a) for a,_,_ in densities.values())==len(plot)
 fig.canvas.draw()
 require_matplotlib_panel_alignment(fig,axes=[ax],panel_ids=['distribution'],json_out=O/'qa'/'overlay.alignment.json',
  overlay_svg=O/'qa'/'overlay.alignment.svg',strict=True)
 name='education_navigation_violin_box_overlay'
 fig.savefig(O/'preview'/f'{name}.png',dpi=300)
 if True:
  fig.savefig(O/f'{name}.pdf',dpi=dpi);fig.savefig(O/f'{name}.svg',dpi=dpi);normalize(O/f'{name}.svg')
  fig.savefig(O/f'{name}.png',dpi=dpi)
 plt.close(fig)
 audit={'scope':'One standalone designated violin figure; no other suite figures regenerated.',
  'source':str(SOURCE),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
  'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  'cohort_n':len(d),'displayed_n':len(plot),'education_or_age_unrecorded_n':int((~known).sum()),
  'displayed_navigation_unrecorded_n':int(plot.navigation.eq('Unrecorded').sum()),
  'cases_displayed':int(plot.observed_dementia_6y.sum()),'boxes':len(summary),
  'single_person_one_point':True,'no_downsampling':True,'bandwidth_years':float(bandwidth),
  'pooled_group_n':{e:len(densities[e][0]) for e in EDU},
  'density_grid_mass':{e:densities[e][2] for e in EDU},
  'notes':['One panel with four category-specific pooled half-violins, not four repeated figures.',
  'Navigation-unrecorded records contribute to pooled density and raw points but not to a fictitious known-navigation box.',
  'Observed dementia is a six-year survey classification, not a confirmed diagnosis at baseline.',
  'Box whiskers are 5th/95th percentiles; all tails remain present in raw observations.',
  'All measurements unchanged; jitter only changes the horizontal display position.',
  'No statistical test, causal estimate, model training or synthetic individual prediction added.']}
 (O/'qa'/'data_and_design_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
 print(json.dumps(audit,ensure_ascii=True,indent=2))
