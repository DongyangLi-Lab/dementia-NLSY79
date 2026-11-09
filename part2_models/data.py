"""Read the frozen Stage 02 archive without changing it."""
import gzip
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARGET = 'outcome_cognition27'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_data(path):
    specification = json.loads((ROOT / 'stage2_config.json').read_text())
    frames = {}
    with ZipFile(path) as archive:
        for cohort in ['HRS', 'NLSY79']:
            name = cohort + '.csv.gz'
            raw = gzip.decompress(archive.read(name))
            require(hashlib.sha256(raw).hexdigest() == specification['output_csv_sha256'][name],
                    'Frozen table checksum mismatch: ' + name)
            frame = pd.read_csv(io.BytesIO(raw))
            require(frame.pair_id.is_unique, 'Duplicate pair identifier')
            require(frame.groupby('household_id').cv_fold5.nunique().eq(1).all(), 'Household split across folds')
            require(frame.age_years.between(48, 55).all(), 'Baseline age outside contract')
            require(frame.followup_years.between(4, 16).all(), 'Duration outside contract')
            require(frame[TARGET].between(0, 27).all(), 'Invalid cognitive outcome')
            require(np.isfinite(frame[TARGET]).all(), 'Nonfinite outcome')
            if cohort == 'NLSY79':
                require(frame.person_id.is_unique, 'NLSY79 must have one row per person')
                require((frame.analysis_role == np.where(frame.cv_fold5.eq(4), 'locked_test', 'development')).all(),
                        'Target split differs from contract')
            frames[cohort] = frame
    return frames, specification


def select_households(frame, per_fold):
    """Deterministic pilot sample; includes every row of each selected household."""
    if per_fold is None:
        return frame.copy()
    selected = frame[['household_id', 'cv_fold5']].drop_duplicates().sort_values('household_id')
    selected = selected.groupby('cv_fold5').head(per_fold).household_id
    return frame.loc[frame.household_id.isin(selected)].reset_index(drop=True)


def pair_weights(frame):
    weights = 1.0 / frame.groupby('person_id').person_id.transform('size').to_numpy()
    return weights / weights.mean()


def categorical_columns():
    rules = json.loads((ROOT / 'metadata/source_rules.json').read_text())
    return rules['types']['nlsy_extension_coded_categorical_candidates']
