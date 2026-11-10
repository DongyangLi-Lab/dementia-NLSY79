"""Create separate output-only result.zip and checkpoint-only model.zip."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
CODE={'.py','.pyc','.mjs','.js','.ps1','.sh','.slurm','.cc','.cpp','.cu','.so'}

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as s:
        for block in iter(lambda:s.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def selected(scope):
    entries=[]
    def tree(source,target):
        for p in sorted(source.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix not in CODE and not p.name.endswith('.inspect.ndjson'):
                entries.append((p,(Path(scope)/target/p.relative_to(source)).as_posix()))
    if scope=='result':
        tree(ROOT/'delivery/tables','tables')
        tree(ROOT/'delivery/historical_reference','historical_figures_and_documents')
        for folder in ['predictions','importance','imputation_comparison','cohorts','figure_inputs']:
            tree(ROOT/'artifacts'/folder,folder)
        rendered=ROOT/'outputs/result_figures'
        report=rendered/'run_report.json'
        if report.exists():
            status=json.loads(report.read_text(encoding='utf-8'))
            required={'tables','fig1','fig2','fig3','fig4','fig5','supplement'}
            if status.get('status')=='complete' and required.issubset(status.get('requested_stages',[])):
                tree(rendered,'figures')
        for name in ['completion_audit.json','source_status.json','verification.json','cohort_verification.json',
                     'comparison_sources_audit.json','final_evaluation_sources_audit.json']:
            entries.append((ROOT/'artifacts'/name,f'result/verification/{name}'))
        entries.append((ROOT/'artifacts/final_shap_sources.json',
                        'result/verification/shap_final_sources.json'))
    else:
        tree(ROOT/'artifacts/checkpoints','checkpoints')
        tree(ROOT/'artifacts/imputation','preprocessing')
        for name in ['prepared_inputs.json','frozen_checkpoint_integrity.json']:
            entries.append((ROOT/'artifacts'/name,f'model/metadata/{name}'))
    return entries

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scope',choices=['result','model'])
    args=parser.parse_args();scope=args.scope
    if scope=='result':
        from prepare_figure_inputs import prepare
        prepare()
    entries=selected(scope);dest=ROOT/'packages'/f'{scope}.zip'
    dest.parent.mkdir(exist_ok=True)
    if scope=='model':
        assert sum(p.suffix=='.pkl' and '/checkpoints/' in n for p,n in entries)==352
        assert sum(p.suffix=='.pkl' and '/preprocessing/' in n for p,n in entries)==44
    else:
        assert not any(p.suffix=='.pkl' or '/unc/' in n or '/checkpoints/' in n for p,n in entries)
    records=[]
    with zipfile.ZipFile(dest.with_suffix('.partial'),'w',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for i,(p,name) in enumerate(entries):
            z.write(p,name)
            records.append({'path':name,'bytes':p.stat().st_size,'sha256':digest(p)})
            if (i+1)%250==0:print(f'{scope}: {i+1}/{len(entries)} files',flush=True)
        z.writestr(f'{scope}/MANIFEST.json',json.dumps(records,indent=2,ensure_ascii=False))
        note=(
            'RESULTS ONLY\nTables include the final three-rule runtime estimates and English-status S1.\n'
            'Predictions, importance/SHAP, imputation comparisons and cohort outputs are retained.\n'
            'figure_inputs contains lightweight saved validation timing and threshold selections, not checkpoints.\n'
            'Rebuild current figures/tables with repository code: python run_figures.py --result result.zip --output generated\n'
            'figures/ (when present) contains the verified result-only regenerated suite and its explicit scope report.\n'
            'historical_figures_and_documents contains earlier presentation outputs, not regenerated revised figures.\n'
            'No model checkpoints, UNC working directories, transfer archives, diagnostic tensors or code are included.\n'
            'Saved verification reports describe the completed pre-cleanup run. UNC diagnostics were subsequently deleted at user request; their replay is no longer available.\n'
            'Individual-level result files are for authorized use and are not published on GitHub.\n'
            if scope=='result' else
            'MODELS ONLY\n352 fitted checkpoints for eight algorithms x two outcomes x two horizons.\n'
            'Includes selected configurations, tuning grids, receipts and 44 preprocessing caches.\n'
            'Model code remains in the repository; no UNC diagnostic artifacts are required in this archive.\n'
            'Only load trusted pickle files in a compatible environment.\n'
        )
        z.writestr(f'{scope}/README.txt',note)
    dest.with_suffix('.partial').replace(dest)
    with zipfile.ZipFile(dest) as z:
        assert z.testzip() is None
        assert len(z.infolist())==len(records)+2
    report={'archive':dest.name,'files':len(records),'bytes':dest.stat().st_size,'sha256':digest(dest),'zip_crc':'passed'}
    (dest.with_suffix('.verification.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    index_path=dest.parent/'INDEX.json'
    index=json.loads(index_path.read_text(encoding='utf-8')) if index_path.exists() else {}
    previous={entry['archive']:entry for entry in index.get('archives',[])}
    previous[dest.name]=report
    index['archives']=[previous[name] for name in ['result.zip','model.zip'] if name in previous]
    index['figure_entry']='python run_figures.py --result packages/result.zip --output outputs/result_figures'
    index_path.write_text(json.dumps(index,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
