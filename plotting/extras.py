"""Output-only tables, four-window participant flow and explanation diagnostics."""
import html
import json
from pathlib import Path
import shutil
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
import common as C


def flow():
    df = C.read(C.RESULT / 'tables/audited_sources/participant_flow.csv')
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    fig.subplots_adjust(left=.05, right=.95, bottom=.04, top=.95, wspace=.20, hspace=.24)
    for ax, (_, r), letter in zip(axes.flat, df.iterrows(), 'ABCD'):
        ax.set_axis_off()
        year = int(r['cohort'].split()[0]); h = int(r['horizon (years)'])
        ax.set_title(f'{year}–{year+h}', loc='center', fontsize=10, pad=10)
        rows = [(f"Eligible at baseline: {int(r['eligible']):,}", C.PURPLE),
                (f"Deaths excluded: {int(r['excluded: died by final wave']):,}\n"
                 f"With prior dementia: {int(r['excluded: death with observed dementia']):,} (subset)", C.GRAY),
                (f"Loss to follow-up: {int(r['excluded: loss to follow-up']):,}\n"
                 f"Unresolved: {int(r['excluded: unresolved after final interview']):,}", C.GRAY),
                (f"Analytic cohort: {int(r['analytic cohort']):,}\n"
                 f"Positive: {int(r['incident dementia']):,}    Negative: {int(r['confirmed non-dementia']):,}", C.WINE)]
        for j, (label, col) in enumerate(rows):
            y = .78-j*.235
            ax.add_patch(FancyBboxPatch((.04,y),.92,.16,boxstyle='round,pad=.012',
                         facecolor='white',edgecolor=col,linewidth=1.2,transform=ax.transAxes))
            ax.text(.5,y+.08,label,ha='center',va='center',fontsize=9,transform=ax.transAxes)
            if j < 3:
                ax.annotate('',xy=(.5,y-.07),xytext=(.5,y-.015),xycoords='axes fraction',
                            arrowprops=dict(arrowstyle='->',color=C.GRAY,lw=.9))
        C.letter(ax, letter)
    C.save(df, 'four_window_participant_flow')
    C.export(fig,'Supplement_outcome_flow',list(axes.flat),list('ABCD'),[['A','B'],['C','D']],[['A','C'],['B','D']])


def explanations():
    p = C.read(C.RESULT / 'tables/audited_sources/importance_permutation_h6.csv')
    s = C.read(C.RESULT / 'tables/audited_sources/importance_shap_h6.csv')
    d = p.merge(s,on=['horizon','model','variable'],suffixes=('_perm','_shap'),validate='one_to_one')
    models = ['LR','GNB','DT','LightGBM','CatBoost','RealMLP','GAT','TabICL']
    fig, axes = plt.subplots(2,4,figsize=(10,5))
    fig.subplots_adjust(left=.07,right=.98,bottom=.12,top=.91,wspace=.48,hspace=.50)
    for ax,m,l in zip(axes.flat,models,'ABCDEFGH'):
        q=d[d.model.eq(m)];assert len(q)==60
        ax.scatter(q.rank_perm,q.rank_shap,s=11,color=C.PURPLE,alpha=.7)
        ax.plot([1,60],[1,60],color=C.GRAY,lw=.7,ls=':')
        ax.set(xlim=(0,61),ylim=(0,61),xticks=[1,30,60],yticks=[1,30,60],xlabel='Permutation rank',ylabel='SHAP rank',title=m)
        C.letter(ax,l)
    C.save(d,'eight_model_explanation_ranks')
    C.export(fig,'Supplement_eight_model_explanation_ranks',list(axes.flat),list('ABCDEFGH'),[list('ABCD'),list('EFGH')])
    # Exact stored per-person contributions; jitter affects display coordinates only.
    source=C.RESULT/'importance/final/h6/TabICL/kernel_shap_values.npz';C.INPUTS.add(source)
    with np.load(source,allow_pickle=False) as z:
        values=z['values'];groups=z['groups']
    assert values.shape==(500,60) and len(groups)==60 and np.isfinite(values).all()
    order=np.argsort(np.mean(np.abs(values),axis=0))[::-1]
    fig,ax=plt.subplots(figsize=(9,12));fig.subplots_adjust(left=.18,right=.96,bottom=.06,top=.98)
    rng=np.random.default_rng(371)
    for j,k in enumerate(order):
        ax.scatter(values[:,k],j+rng.uniform(-.28,.28,len(values)),s=1.2,c=C.PURPLE,alpha=.32,rasterized=True,linewidths=0)
    ax.axvline(0,color=C.GRAY,lw=.6);ax.set(yticks=np.arange(60),yticklabels=groups[order],ylim=(59.8,-.8),xlabel='KernelSHAP contribution (saved model output scale)')
    C.save(pd.DataFrame(values,columns=groups),'TabICL_500_person_SHAP')
    C.export(fig,'Supplement_TabICL_500_person_SHAP',[ax],['A'])


def tables():
    """Keep exact CSV text, paginate by rows/columns, repeat identifiers on each page."""
    sources=C.RESULT/'tables/audited_sources/supplementary'
    target=C.O/'tables';target.mkdir(exist_ok=True)
    index=[]
    for source in sorted(sources.glob('*.csv')):
        print(f'  exporting {source.stem}',flush=True)
        d=C.read(source).fillna('')
        # Read strings separately: no re-rounding of submitted estimates/CI strings.
        d=pd.read_csv(source,dtype=str,keep_default_na=False)
        shutil.copy2(source,target/source.name)
        title='Table 1 — Model performance' if source.stem=='Table1' else f'Table {source.stem}'
        display=d.copy()
        if source.stem=='S1':
            display['Classification']=display['Classification'].replace({'Y = 1':'Positive','Y = 0':'Negative'})
            display.loc[display.Category.eq('Eligible at baseline'),'Classification']='Baseline'
            display.loc[display.Category.eq('Analytic cohort'),'Classification']='Positive or negative'
            display['Category']=display.Category.map(lambda s:'\u2003'+s.strip() if s.startswith('  ') else s)
            display=display.drop(columns=['Row type'])
        body=display.to_html(index=False,escape=True,border=0)
        (target/f'{source.stem}.html').write_text('<!doctype html><meta charset="utf-8"><style>'
            'body{font:14px Georgia,serif;margin:32px;color:#222}table{border-collapse:collapse;width:100%;border-top:2px solid #111;border-bottom:2px solid #111}'
            'th{border-bottom:1px solid #111;text-align:left}td,th{padding:7px 10px;vertical-align:top}tbody tr:nth-child(even){background:#fafafa}'
            '@media print{thead{display:table-header-group}tr{break-inside:avoid}}</style>'
            f'<h2>{html.escape(title)}</h2>'+body,encoding='utf-8')
        # Broad tables are split into readable blocks; all values are retained.
        columns=list(d.columns);keys=columns[:min(3,len(columns))]
        if 'level' in columns and 'level' not in keys:keys.append('level')
        rest=[c for c in columns if c not in keys];width=max(1,7-len(keys))
        blocks=[keys+rest[i:i+width] for i in range(0,len(rest),width)] or [keys]
        if source.stem=='S1':
            shutil.copy2(C.RESULT/'tables/outcome_flow/Supplementary_Table_S1_English_status.pdf',target/'S1.pdf')
            index.append(dict(table=source.stem,rows=len(d),columns=len(d.columns)))
            continue
        with PdfPages(target/f'{source.stem}.pdf') as pdf:
            for cols in blocks:
                wrapped=[[textwrap.fill(str(v),width=18,break_long_words=True) for v in row]
                         for row in d[cols].to_numpy()]
                start=0
                while start<len(wrapped):
                    rows=[];units=0
                    while start<len(wrapped):
                        row=wrapped[start];height=max(v.count('\n')+1 for v in row)+1
                        if rows and units+height>42:break
                        rows.append(row);units+=height;start+=1
                    fig,ax=plt.subplots(figsize=(11.7,8.3));ax.set_axis_off()
                    fig.subplots_adjust(left=.035,right=.965,bottom=.06,top=.91)
                    head=[textwrap.fill(c,18) for c in cols]
                    table=ax.table(cellText=rows,colLabels=head,cellLoc='left',colLoc='left',loc='upper center')
                    table.auto_set_font_size(False);table.set_fontsize(7.5)
                    heights=[max(c.count('\n')+1 for c in head)+1]+[max(c.count('\n')+1 for c in row)+1 for row in rows]
                    for (r,c),cell in table.get_celld().items():
                        cell.set_height(heights[r]*.018)
                        cell.set_edgecolor('#222');cell.set_linewidth(.5)
                        cell.visible_edges='TB' if r==0 else 'B' if r==len(rows) else ''
                        cell.set_text_props(fontfamily='DejaVu Serif',weight='bold' if r==0 else 'normal')
                    fig.text(.035,.955,title,fontsize=12,fontfamily='DejaVu Serif')
                    fig.text(.035,.025,f'Rows {start-len(rows)+1}–{start} of {len(d)}; exact source: {source.name}',fontsize=8)
                    pdf.savefig(fig);plt.close(fig)
        index.append(dict(table=source.stem,rows=len(d),columns=len(d.columns)))
    # These previously approved outputs are retained byte-for-byte, not re-authored.
    for name in ['runtime','outcome_flow']:
        source=C.RESULT/'tables'/name
        if source.is_dir():shutil.copytree(source,target/name,dirs_exist_ok=True)
    (target/'index.json').write_text(json.dumps(index,indent=2),encoding='utf-8')
    return index
