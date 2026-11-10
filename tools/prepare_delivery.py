"""Collect aggregate results and approved historical presentation references."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'delivery'

def copy(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)
    assert hashlib.sha256(source.read_bytes()).digest()==hashlib.sha256(target.read_bytes()).digest()

def main():
    for p in (ROOT/'artifacts/tables').rglob('*.csv'):
        if 'supplementary_partial_verified' not in p.parts:
            copy(p,OUT/'tables/audited_sources'/p.relative_to(ROOT/'artifacts/tables'))
    for name in ['completion_audit.json','source_status.json','verification.json','runtime_completion_audit.json',
                 'comparison_sources_audit.json','final_evaluation_sources_audit.json',
                 'tabicl_cross_device_acceptance.json','tabicl_permutation_execution_audit.json']:
        copy(ROOT/'artifacts'/name,OUT/'verification'/name)
    old=ROOT.parent/'output'
    for name in ['ACTIVE_SUITE.json','ACTIVE_REFERENCE_STORY.json','ACTIVE_TABLES.json','ACTIVE_FIG5.json']:
        copy(old/name,OUT/'historical_reference/indexes'/name)
    bundle=old/'manuscript_bundle_v2_no_ci'
    for p in bundle.rglob('*'):
        if p.is_file() and (p.parent==bundle and p.suffix=='.docx' or p.relative_to(bundle).parts[0] in ['figures','captions']):
            copy(p,OUT/'historical_reference/selected_figure_suite'/p.relative_to(bundle))
    for folder in ['research_tables_main_ci_v3','research_tables_main_supp_v2','reference_pairs_word_v1']:
        for p in (old/folder).glob('*.docx'):
            copy(p,OUT/'historical_reference/documents'/p.name)
    for p in (old/'shap_validation_all8_v14/two_options_word').glob('*.docx'):
        copy(p,OUT/'historical_reference/documents'/p.name)
    records=[]
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p.name!='MANIFEST.json' and not p.name.endswith('.inspect.ndjson'):
            records.append({'path':p.relative_to(OUT).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    (OUT/'MANIFEST.json').write_text(json.dumps({'files':records},indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'Collected and hash-checked {len(records)} files')

if __name__=='__main__':main()
