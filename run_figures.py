"""Rebuild the current five-figure suite and tables from result/ or result.zip.

No training, imputation, model checkpoint, UNC, or historical-table fallback.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parent
PLOT = ROOT / 'plotting'
STAGES = ['tables', 'fig1', 'fig2', 'fig3', 'fig4', 'fig5', 'supplement']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def resolve_result(path, scratch):
    path=path.resolve()
    if path.is_file():
        if not zipfile.is_zipfile(path):raise ValueError('Expected result.zip or an extracted result directory')
        with zipfile.ZipFile(path) as z:
            for item in z.infolist():
                dest=(scratch/item.filename).resolve()
                if not dest.is_relative_to(scratch.resolve()):raise ValueError('Unsafe ZIP member')
                if item.filename.startswith('/') or '\\' in item.filename:raise ValueError('Unsafe ZIP path')
            z.extractall(scratch)
        path=scratch/'result'
    elif (path/'result').is_dir():path=path/'result'
    if not (path/'tables/audited_sources').is_dir():
        raise ValueError('Missing result/tables/audited_sources; historical results/tables is not accepted')
    manifest=path/'MANIFEST.json'
    records=[]
    if manifest.exists():
        for row in json.loads(manifest.read_text(encoding='utf-8')):
            name=row['path']
            if not name.startswith('result/'):raise ValueError('Expected a result manifest')
            source=(path/name.removeprefix('result/')).resolve()
            if not source.is_relative_to(path.resolve()):raise ValueError('Unsafe manifest path')
            if not source.is_file() or sha(source)!=row['sha256']:
                raise ValueError(f'Input checksum mismatch: {name}')
            records.append(row)
    else:
        records=[{'path':p.relative_to(path).as_posix(),'sha256':sha(p)}
                 for p in sorted(path.rglob('*')) if p.is_file()]
    return path,records


def captions(out, checks, samples):
    counts={r['window']:r for r in checks}
    a,b=counts['2000-2006'],counts['2010-2016']
    common_en=('All cohorts use the revised outcome: observed follow-up dementia is positive only among confirmed final-wave survivors; '
               'negative outcomes require final-wave non-dementia. Deaths are excluded in both primary and sensitivity analyses. '
               'No refitting or new individual predictions. Graphical confidence intervals appear only in Fig5 C/D. ')
    common_zh='全部队列使用修订结局：仅末次波次确认存活者可纳入，阳性须在随访观察到痴呆，阴性须在末次波次确认无痴呆；主分析和敏感性分析均排除死亡。本次不重新训练或生成个体预测；图形置信区间仅出现在Fig5 C/D。'
    entries={
      'Fig1':(f'Observation paths and age distribution. A/B: 2000 and 2010 six-year survivor cohorts, n={a["n"]:,} and {b["n"]:,}. '
               f'C/D: first observed cases, {a["cases"]:,} and {b["cases"]:,}. E: all observed sequence frequencies. F: cumulative case share ordered by baseline age. '
               'N/D/? indicate observed non-dementia/dementia/unavailable assessment. Missing assessment is not assigned death. See Supplement_outcome_flow for all four baseline-to-analysis flows.',
               f'观察路径与首次痴呆观察的年龄分布。2000、2010六年队列分别为{a["n"]:,}、{b["n"]:,}人，阳性{a["cases"]:,}、{b["cases"]:,}人。A/B为观察路径，C/D为首次观察病例年龄，E为全部路径频数，F为按基线年龄排序的累积病例份额。N/D/?表示无痴呆/痴呆/评估缺失，评估缺失不等于死亡；四窗口完整排除流程见Supplement_outcome_flow。'),
      'Fig2':('Interview coverage, joint age/height/weight distributions, missing inputs, visits and ADL outcome mass in the revised 2010 six-year cohort. '
              'Contours enclose 50%/80% KDE probability mass, not confidence intervals. Every eligible joint observation is used; unavailable coordinates are enumerated in source_data. '
              'The lower panel includes all 60 grouped predictors across four cohorts.',
              '修订后2010六年队列的访谈覆盖、年龄/身高/体重联合分布、输入缺失、就医次数和ADL结局构成。轮廓为KDE的50%/80%概率质量，不是置信区间；全部具备联合坐标的记录参与绘图，缺失数量保存在来源表。下方包含四队列全部60项预测因素。'),
      'Fig3':('Age/BMI distributions by map-use state, using all observations with available coordinates. Unknown map-use status is recorded separately. '
              'Panel G preserves the selected v3 pooled half-violin/grouped-box/raw-point layout, refreshed from the revised cohort. '
              'Boxes show quartiles and 5th/95th percentile whiskers; triangles are means. Jitter changes display x coordinates only. No other main figure uses violins.',
              '按地图使用状态展示年龄/BMI分布，使用全部坐标可用记录，未知地图状态单独计数。G保持已选v3的合并半小提琴、分组箱线和全部散点布局，数据更新为修订队列。箱体为四分位数、须为5/95百分位、三角为均值；抖动只影响横向显示坐标。只有本图使用小提琴结构。'),
      'Fig4':('All eight verified models and all 60 predictors: permutation ranks, cross-model dispersion, paired SHAP/permutation ranks and importance point estimates. '
              '480 paired ranks are included. Importance is model reliance, not a causal effect. SHAP methods/output scales differ by model; raw SHAP magnitudes must not be compared between algorithms. '
              'TabICL uses the formal 500-person approximate KernelSHAP output; numerical approximation and recorded cross-device acceptance remain relevant.',
              '使用全部八个已核验模型和60项预测因素：置换排名、跨模型分散度、SHAP/置换配对排名及重要性点估计，共480对排名。重要性反映模型依赖，不代表因果效应。各模型SHAP方法及输出尺度不同，不能直接跨算法比较原始SHAP绝对值。TabICL使用正式500人近似KernelSHAP结果，仍须保留近似计算和跨设备验收边界。'),
      'Fig5':(f'A/B: saved revised calibration residuals; point area represents bin share. C/D: TabICL minus LR/CatBoost net benefit per 1,000, '
               f'79 thresholds from 0.01 to 0.40; pointwise 95% percentile intervals from {samples:,} shared household-cluster bootstrap draws, fixed fitted models, '
               'no simultaneous coverage or training variability. Cobalt/vermilion bands have opacity 0.24. E/F: saved validation-fit durations and selected thresholds. '
               'All panels now derive from the same revised benchmark. These are descriptive diagnostics, not validated clinical utility.',
               f'A/B为本轮校准残差，点面积表示分箱占比。C/D为TabICL相对LR/CatBoost每千人的净获益差，展示0.01至0.40的79个固定阈值；用{samples:,}次共享家庭聚类bootstrap计算逐点95%百分位区间，条件于固定模型，不含训练变异或同时覆盖。钴蓝/朱红置信带透明度为0.24。E/F为保存的调参拟合时间与选定阈值；全部面板现在均来自同一轮修订分析。这些是描述性诊断，不构成已验证的临床效用。')}
    for name,(en,zh) in entries.items():
        (out/f'captions/{name}_en.txt').write_text(en+'\n\n'+common_en,encoding='utf-8')
        (out/f'captions/{name}_zh.txt').write_text(zh+'\n\n'+common_zh,encoding='utf-8')
    (out/'captions/Supplement_en.txt').write_text(
        'Outcome flow uses the audited four-window counts; indented death-with-dementia counts are subsets. '
        'Explanation-rank panels retain 60 predictors per model. The TabICL contribution display retains all 500 × 60 stored contributions, '
        'orders predictors by mean absolute contribution and uses display-only vertical jitter. No feature-value color or synthetic observations are introduced.',encoding='utf-8')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result',type=Path,required=True,help='result.zip, extracted result/, or its parent')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/result_figures')
    parser.add_argument('--only',default='all',help='Comma-separated: '+','.join(STAGES))
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--bootstrap',type=int,default=2000)
    parser.add_argument('--dpi',type=int,default=300)
    parser.add_argument('--overwrite',action='store_true',help='Keep existing output as a timestamped previous directory')
    args=parser.parse_args(argv)
    chosen=STAGES if args.only=='all' else args.only.split(',')
    if set(chosen)-set(STAGES):parser.error('Unknown stage')
    if args.bootstrap<1000:parser.error('Use at least 1000 bootstrap draws; default is 2000')
    if args.dpi<150:parser.error('Use at least 150 dpi; 300 default, 600 for submission export')
    out=args.output.resolve();input_path=args.result.resolve()
    if out==input_path or (input_path.is_dir() and out.is_relative_to(input_path)):
        parser.error('Output must be outside the immutable input result directory')
    if out.exists() and any(out.iterdir()):
        if not args.overwrite:parser.error('Output exists; use another directory or --overwrite (preserves previous output)')
        previous=out.with_name(out.name+'.previous_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        out.rename(previous)
    out.mkdir(parents=True,exist_ok=True)
    report={'status':'running','stages':{},'input':str(input_path),'output':str(out),'created_utc':datetime.now(timezone.utc).isoformat(),
            'scope':'Five current main figures, three supporting figures, all 23 saved numbered tables. Historical P/M/core composites and original S2-S9 numbering are not relabeled as revised.',
            'bootstrap':args.bootstrap,'dpi':args.dpi}
    def checkpoint():
        (out/'run_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    def stage(name,call):
        start=time.monotonic();print(f'[{name}] running',flush=True)
        try:
            call();report['stages'][name]={'status':'complete','seconds':round(time.monotonic()-start,2)}
        except Exception as e:
            report['stages'][name]={'status':'failed','error':str(e)};checkpoint();raise
        checkpoint();print(f'[{name}] complete',flush=True)
    checkpoint()
    try:
        # Limit BLAS parallelism for predictable, CPU-only rendering.
        for key in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:
            os.environ.setdefault(key,'1')
        sys.path.insert(0,str(PLOT))
        import common as C
        from prepare import prepare
        import extras
        with tempfile.TemporaryDirectory(prefix='hrs_result_') as temp:
            result,records=resolve_result(input_path,Path(temp))
            report['input_manifest_sha256']=hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()
            report['code_sha256']={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(PLOT.glob('*.py'))}
            report['code_sha256']['run_figures.py']=sha(Path(__file__))
            C.O=C.OUT=out;C.RESULT=result;C.PREP=out/'input_tables';C.DPI=args.dpi;C.FINAL=True
            checks=prepare(result,out)
            report['cohorts']=checks
            if args.check_only:
                report['status']='validated_inputs';checkpoint();return 0
            import suite
            if 'tables' in chosen:stage('tables',extras.tables)
            for n in range(1,6):
                if f'fig{n}' not in chosen:continue
                if n==3:
                    from overlay import render
                    stage('fig3_component',lambda:render(result/'cohorts/2010-2016_final.csv',out/'selected_component',args.dpi))
                if n==5:
                    from reliability import run
                    stage('fig5_intervals',lambda:run(result,out/'source_data',args.bootstrap))
                stage(f'fig{n}',getattr(suite,f'fig{n}'))
            if 'supplement' in chosen:
                stage('supplement_flow',extras.flow)
                stage('supplement_explanations',extras.explanations)
            captions(out,checks,args.bootstrap)
            C.manifest()
            # Input identities use archive-relative paths, never the temporary extraction path.
            used={p.relative_to(result).as_posix():sha(p) for p in C.INPUTS if p.is_relative_to(result)}
            report['used_result_sources']=used
            report['requested_stages']=chosen
            report['unregenerated_historical_items']=['Figure P population embeddings','Figure M model-response probes','core combined_story',
                'Original manuscript supplementary-figure S2-S9 numbering requires a separate figure-to-source mapping; current supports use descriptive names.']
            qa=[]
            for pdf in sorted(out.glob('*.pdf'))+sorted((out/'selected_component').glob('*.pdf')):
                for tool,extra in [('audit_pdf_text.py',['--min-pt','5','--json']),
                                   ('audit_figure_collisions.py',['--json-out',str(out/'qa'/f'{pdf.stem}.collision.json')])]:
                    proc=subprocess.run([sys.executable,'-X','utf8',str(PLOT/tool),str(pdf),*extra],capture_output=True,text=True,encoding='utf-8',errors='replace')
                    (out/'qa'/f'{pdf.stem}.{tool}.txt').write_text(proc.stdout+proc.stderr,encoding='utf-8')
                    qa.append({'file':str(pdf.relative_to(out)),'audit':tool,'exit_code':proc.returncode})
            report['rendered_qa']=qa
            report['status']='complete' if all(q['exit_code']==0 for q in qa) else 'rendered_needs_qa_fixes'
            # Browsable index: figures first, then exact source tables and bilingual captions.
            links=''.join(f'<li><a href="{html.escape(p.relative_to(out).as_posix())}">{html.escape(p.stem)}</a></li>' for p in sorted(out.glob('*.pdf')))
            imgs=''.join(f'<h2>{html.escape(p.stem)}</h2><a href="{p.with_suffix(".pdf").name}"><img src="{p.name}" style="max-width:100%"></a>' for p in sorted(out.glob('Fig*.png')) if 'component' not in p.stem)
            ts=''.join(f'<li><a href="tables/{p.name}">{p.stem}</a></li>' for p in sorted((out/'tables').glob('*.html')))
            (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>body{max-width:1100px;margin:32px auto;font:16px Arial;color:#25252b}h1,h2{color:#553574}</style>'
                '<h1>HRS revised-result figures and tables</h1><p>Generated from revised result inputs. See run_report.json for scope, checks and remaining historical items.</p>'
                '<ul>'+links+'</ul>'+imgs+'<h2>Tables</h2><ul>'+ts+'</ul>',encoding='utf-8')
            report['outputs']=[{'path':p.relative_to(out).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)}
                               for p in sorted(out.rglob('*')) if p.is_file() and p.name!='run_report.json']
            checkpoint()
            print(f'{report["status"]}: {out}',flush=True)
            return 0 if report['status']=='complete' else 1
    except Exception as e:
        report['status']='failed';report['error']=str(e);checkpoint();traceback.print_exc();return 1


if __name__=='__main__':
    raise SystemExit(main())
