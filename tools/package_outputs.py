"""Package saved scientific outputs without rerunning analysis.

Local archives may contain HRS-derived participant data and model checkpoints.
They are deliberately ignored by Git. Public delivery/ contains aggregate data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
SKIP={'__pycache__','node_modules','runtime_hardware_estimate_v1'}
SOURCE_EXT={'.py','.pyc','.mjs','.js','.ps1','.sh','.slurm','.cu','.cpp','.c','.h'}

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def archive(source,target,exclude_code=False,prefix=''):
    files=sorted(p for p in source.rglob('*') if p.is_file() and not SKIP.intersection(p.relative_to(source).parts)
                 and not (exclude_code and p.suffix.lower() in SOURCE_EXT))
    target.parent.mkdir(parents=True,exist_ok=True)
    records=[]
    with zipfile.ZipFile(target.with_suffix('.partial'), 'w',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for i,p in enumerate(files):
            name=(Path(prefix)/p.relative_to(source)).as_posix()
            z.write(p,name)
            records.append({'path':name,'bytes':p.stat().st_size,'sha256':digest(p)})
            if (i+1)%250==0: print(f'{target.name}: {i+1}/{len(files)}',flush=True)
        z.writestr('ARCHIVE_MANIFEST.json',json.dumps(records,indent=2))
        z.writestr('README.txt', 'Saved outputs only. Current revised results and historical outputs are separate.\nHistorical outputs include rejected/superseded designs and must not be treated as current results.\nHRS-derived individual records and checkpoints are for authorized local use, not GitHub publication.\n')
    target.with_suffix('.partial').replace(target)
    with zipfile.ZipFile(target) as z:
        bad=z.testzip()
        if bad: raise ValueError(f'Corrupt member: {bad}')
        assert len(z.infolist())==len(records)+2
    report={'archive':target.name,'files':len(records),'bytes':target.stat().st_size,'sha256':digest(target),'zip_crc':'passed'}
    target.with_suffix('.verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scope',choices=['revised','history'])
    args=parser.parse_args()
    if args.scope=='revised':archive(ROOT/'artifacts',ROOT/'packages/HRS_revised_private_outputs_20261110.zip',True,'artifacts')
    else:archive(ROOT.parent/'output',ROOT/'packages/HRS_historical_outputs_20261110.zip',False,'historical_outputs_NOT_CURRENT')

if __name__=='__main__':main()
