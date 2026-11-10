"""Read-only verification of the delivery against saved final result evidence."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'delivery'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)

def main():
    status=json.loads((ROOT/'artifacts/source_status.json').read_text())
    require(status['state']=='complete','Final source status is incomplete')
    for name,record in status['final_tables'].items():
        src=ROOT/'artifacts/tables/supplementary'/f'{name}.csv'
        require(sha(src)==record['sha256'],f'Audited source changed: {name}')
        require(sha(src)==sha(D/'tables/audited_sources/supplementary'/src.name),f'Delivery differs: {name}')
    manifest=json.loads((D/'MANIFEST.json').read_text(encoding='utf-8'))
    for rec in manifest['files']:
        require(sha(D/rec['path'])==rec['sha256'],f'Delivery hash mismatch: {rec["path"]}')
    original=list(csv.DictReader((ROOT/'artifacts/tables/supplementary/S1.csv').open(encoding='utf-8-sig')))
    displayed=list(csv.DictReader((D/'tables/outcome_flow/Supplementary_Table_S1_English_status.csv').open(encoding='utf-8-sig')))
    for a,b in zip(original,displayed,strict=True):
        require(all(a[k]==b[k] for k in ['Category','Row type','2000-2006','2000-2008','2010-2016','2010-2018']),'S1 numeric/subset information changed')
        require(b['Classification'] in ['Baseline','Positive or negative','Positive','Negative','Excluded'],'Non-English status')
    measured=[]
    for name in ['runtime.csv','sensitivity_any_runtime.csv']:
        measured+=list(csv.DictReader((ROOT/'artifacts/tables'/name).open(encoding='utf-8')))
    summary=list(csv.DictReader((D/'tables/runtime/runtime_summary.csv').open(encoding='utf-8')))
    totals=[0.,0.,0.]
    for row in summary[:-1]:
        group=[r for r in measured if r['model']==row['Model']]
        require(len(group)==4,'Runtime scope mismatch')
        fit=sum(float(r['total_minutes']) for r in group);pred=sum(float(r['total_prediction_minutes']) for r in group)
        for col,want in [('Measured_fit_min',fit),('Measured_prediction_min',pred),('Measured_total_min',fit+pred)]:
            require(abs(float(row[col])-want)<1e-9,f'Runtime value changed: {row["Model"]}/{col}')
        for i,v in enumerate([fit,pred,fit+pred]):totals[i]+=v
    for col,want in zip(['Measured_fit_min','Measured_prediction_min','Measured_total_min'],totals):
        require(abs(float(summary[-1][col])-want)<1e-9,'Runtime total mismatch')
    require(not list((D/'tables/runtime').glob('*.py')) and not list((D/'tables/runtime').glob('*.tex')),'Runtime render code in delivery')
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    source_count=0
    for name in tracked:
        p=ROOT/name
        if name.endswith('.py') and p.is_file():
            ast.parse(p.read_text(encoding='utf-8-sig'),filename=name);source_count+=1
    print(json.dumps({'status':'passed','numbered_tables':len(status['final_tables']),'delivery_files':len(manifest['files']),
                      'S1_rows':len(displayed),'runtime_combinations':len(measured),'runtime_total_minutes':totals[-1],
                      'python_sources_parsed':source_count},indent=2))

if __name__=='__main__':main()
