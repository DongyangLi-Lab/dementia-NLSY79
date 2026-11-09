"""Build the model-ready cohort archive from the harmonized cohort archive."""

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile


HERE = Path(__file__).resolve().parent
TABLES = 'data/processed/hrs_complete_v3/'
BRANCH = 'strict27_matched_age_actual4_16.csv.gz'
README = '''# Model-ready cohort pairs

HRS.csv.gz contains 52,023 baseline-to-follow-up pairs from 8,299 participants.
NLSY79.csv.gz contains one pair per participant for 2,305 participants, with
82 additional life-course candidate predictors already joined by person ID.

The analysis uses baseline ages 48-55, actual follow-up intervals of 4-16
years, and strictly observed 27-point cognition scores. Use followup_years
as a predictor and outcome_cognition27 as the target. Use followup_band only
for reporting (4 to under 8, 8 to under 12, and 12 through 16 years).

Use the feature lists in features.json to select inputs. Preserve household_id
and cv_fold5 when splitting the data. NLSY79 analysis_role identifies 1,845
development and 460 locked-test participants. For HRS, assign each pair a
relative weight of one divided by the number of eligible pairs for that person
in the training fold, then normalize within the fold. The flag
is_first_eligible_pair selects the earliest eligible pair per participant.
'''


def read_table(archive, name, expected_hash):
    payload = archive.read(TABLES + name)
    if hashlib.sha256(payload).hexdigest() != expected_hash:
        raise ValueError(f'Unexpected source content: {name}')
    stream = io.StringIO(gzip.decompress(payload).decode('utf-8-sig'), newline='')
    reader = csv.DictReader(stream)
    return reader.fieldnames, list(reader)


def followup_band(years):
    duration = float(years)
    if 4 <= duration < 8:
        return '4_to_lt8'
    if 8 <= duration < 12:
        return '8_to_lt12'
    if 12 <= duration <= 16:
        return '12_to_16'
    raise ValueError('Follow-up duration is outside the analysis window.')


def write_table(archive, name, columns, rows, expected_hash):
    text = io.StringIO(newline='')
    writer = csv.DictWriter(text, fieldnames=columns, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    payload = text.getvalue().encode('utf-8')
    if hashlib.sha256(payload).hexdigest() != expected_hash:
        raise ValueError(f'Generated CSV differs from the validated export: {name}')
    compressed = gzip.compress(payload, compresslevel=9, mtime=0)
    archive.writestr(name, compressed)
    return hashlib.sha256(compressed).hexdigest()


def build(source, destination):
    spec = json.loads((HERE / 'stage2_config.json').read_text(encoding='utf-8'))
    hashes = spec['input_sha256']
    with ZipFile(source) as archive:
        hrs_name = 'hrs_analysis_allpairs_' + BRANCH
        first_name = 'hrs_analysis_' + BRANCH
        nlsy_name = 'nlsy_analysis_' + BRANCH
        extensions_name = 'nlsy_lifecourse_extensions.csv.gz'
        hrs_columns, hrs = read_table(archive, hrs_name, hashes[hrs_name])
        _, first = read_table(archive, first_name, hashes[first_name])
        nlsy_columns, nlsy = read_table(archive, nlsy_name, hashes[nlsy_name])
        _, extensions = read_table(archive, extensions_name, hashes[extensions_name])

    first_ids = {row['pair_id'] for row in first}
    extensions_by_person = {row['person_id']: row for row in extensions}
    additional = spec['features']['feature_sets']['nlsy_additional82']
    for row in hrs:
        row['followup_band'] = followup_band(row['followup_years'])
        row['is_first_eligible_pair'] = str(int(row['pair_id'] in first_ids))
    for row in nlsy:
        extra = extensions_by_person[row['person_id']]
        row.update({column: extra[column] for column in additional})
        row['followup_band'] = followup_band(row['followup_years'])
        row['analysis_role'] = 'locked_test' if float(row['cv_fold5']) == 4 else 'development'

    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as temporary:
        pending = Path(temporary) / destination.name
        with ZipFile(pending, 'w', compression=ZIP_DEFLATED) as archive:
            expected = spec['output_csv_sha256']
            hrs_hash = write_table(archive, 'HRS.csv.gz', hrs_columns + ['followup_band', 'is_first_eligible_pair'], hrs, expected['HRS.csv.gz'])
            nlsy_hash = write_table(archive, 'NLSY79.csv.gz', nlsy_columns + additional + ['followup_band', 'analysis_role'], nlsy, expected['NLSY79.csv.gz'])
            features = dict(spec['features'], output_sha256={'HRS.csv.gz': hrs_hash, 'NLSY79.csv.gz': nlsy_hash})
            archive.writestr('features.json', json.dumps(features, indent=2) + '\n')
            archive.writestr('README.md', README)
        pending.rename(destination)
    print(f'Wrote {destination} with {len(hrs):,} HRS pairs and {len(nlsy):,} NLSY79 pairs.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='Stage 01 ZIP archive')
    parser.add_argument('--output', type=Path, default=HERE / 'local_data/02_model_ready_cohort_pairs_v1.zip')
    args = parser.parse_args()
    build(args.source, args.output)


if __name__ == '__main__':
    main()
