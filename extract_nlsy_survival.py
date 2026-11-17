"""Extract official respondent mortality and interview fields without changing cohorts."""
from pathlib import Path
from zipfile import ZipFile
import hashlib
import json
import re
import time
import argparse

import pandas as pd

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def extract(source, catalog_path, dest, report):
    catalog = pd.read_csv(catalog_path, dtype=str, keep_default_na=False)
    selected = catalog.loc[catalog.question.eq('RNI') | catalog.column.isin(
        ['R0000100', 'N0022000', 'T8780600', 'T8787500', 'T8330001', 'T8330002'])].copy()
    dest.parent.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    selected.to_csv(report/'nlsy_survival_source_fields.csv', index=False)
    start = time.monotonic()
    with ZipFile(source) as archive:
        text = archive.read(next(n for n in archive.namelist() if n.endswith('.cdb'))).decode('utf-8')
        starts = list(re.finditer(r'(?m)^([A-Z]\d{5}\.\d{2})\s+\[', text))
        refs = set(selected.ref)
        sections = ''.join(text[m.start():starts[i+1].start() if i+1<len(starts) else len(text)]
                           for i,m in enumerate(starts) if m.group(1) in refs)
        (report/'nlsy_survival_selected_codebook.cdb').write_text(sections, encoding='utf-8')
        with archive.open(next(n for n in archive.namelist() if n.endswith('.csv'))) as stream:
            chunks = []
            for chunk in pd.read_csv(stream, usecols=selected.column.tolist(), dtype='float64', chunksize=1024):
                chunks.append(chunk)
                print('Extracted', sum(len(x) for x in chunks), 'respondents;', round(time.monotonic()-start,1), 'seconds', flush=True)
    data = pd.concat(chunks, ignore_index=True)
    assert len(data)==12686 and data.R0000100.is_unique and data.R0000100.notna().all()
    data.to_csv(dest, index=False, compression='gzip')
    metadata = {'source_zip_sha256': digest(source), 'extract_sha256': digest(dest), 'rows':len(data),
                'columns':selected.column.tolist(), 'rni_by_year':dict(zip(selected.loc[selected.question.eq('RNI'),'year'],selected.loc[selected.question.eq('RNI'),'column'])),
                'source_csv_read_to_eof_crc_checked':True,
                'codebook_sha256':digest(report/'nlsy_survival_selected_codebook.cdb')}
    (report/'nlsy_survival_extract.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    print('Completed mortality extract',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['source', 'catalog', 'output', 'reports']:
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    extract(args.source, args.catalog, args.output, args.reports)
