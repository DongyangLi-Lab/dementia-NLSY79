"""Archive tracked source and delivery files; omit raw data and BASE results."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[1]
CODE={'.py','.cc','.cu','.h','.slurm','.sh','.ps1','.ipynb','.R','.md','.txt','.toml','.yaml','.yml'}

def main():
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    paths=[]
    for name in tracked:
        if not name:continue
        p=ROOT/name;parts=Path(name).parts
        if parts[0] in {'data','results','figures','outputs','tmp','packages'}:continue
        if parts[0]=='delivery' or p.suffix in CODE or p.name in {'.gitattributes','.gitignore','build_evidence.json'}:
            if p.is_file():paths.append(p)
    target=ROOT/'packages/HRS_code_and_delivery_20261110.zip';target.parent.mkdir(exist_ok=True)
    manifest=[]
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(paths):
            name=p.relative_to(ROOT).as_posix();content=p.read_bytes()
            z.writestr(name,content);manifest.append({'path':name,'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()})
        z.writestr('CODE_AND_DELIVERY_MANIFEST.json',json.dumps(manifest,indent=2,ensure_ascii=False))
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        assert len(z.infolist())==len(manifest)+1
        assert not any('runtime_hardware_estimate' in n or n.startswith('tmp/') for n in z.namelist())
        assert 'delivery/tables/runtime/HRS_runtime.pdf' in z.namelist()
        assert 'delivery/tables/outcome_flow/Supplementary_Table_S1_English_status.pdf' in z.namelist()
        assert 'delivery/historical_reference/english_status_S1/Supplementary_Table_S1_English_status.pdf' in z.namelist()
    report={'archive':target.name,'files':len(manifest),'bytes':target.stat().st_size,
            'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'zip_crc':'passed'}
    target.with_suffix('.verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))

if __name__=='__main__':main()
