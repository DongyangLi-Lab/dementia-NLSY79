"""Descriptive tables for the analytic cohorts: participant flow, complete vs incomplete records, and the overlap
between the 2000 development and 2010 temporal-validation cohorts. Writes aggregate tables to results/tables/.

Usage: python cohort_tables.py
"""
import os

import numpy as np
import pandas as pd

from inference import evaluation
from train.cohorts import ARTIFACTS, atomic_csv, load_cohorts
from model.specs import MODELS

OUT = ARTIFACTS / "tables"
HORIZONS = (6, 8)


def household(hhidpn):
    return np.asarray(hhidpn) // 1000  # hhidpn = HHID * 1000 + PN


def overlap_status(temporal_ids, development_ids):
    """Per temporal-validation participant: returning participant / new participant in a development household / new."""
    t = np.asarray(temporal_ids)
    dev = set(development_ids)
    dev_hh = set(household(list(dev)))
    return np.where(np.isin(t, list(dev)), 'returning participant',
                    np.where(np.isin(household(t), list(dev_hh)), 'new participant, development household',
                             'new participant and household'))


def _smd(a, b):
    a, b = a.dropna(), b.dropna()
    pooled = np.sqrt((a.var() + b.var()) / 2)
    return (a.mean() - b.mean()) / pooled if pooled > 0 else np.nan


def complete_vs_incomplete():
    characteristics = {
        'incident dementia (%)': lambda d: d['y'] * 100,
        'age (years)': lambda d: d['ageym'],
        'female (%)': lambda d: d['gender'] * 100,
        'proxy interview (%)': lambda d: d['proxy'] * 100,
        'non-Hispanic Black (%)': lambda d: d['raceeth1 '] * 100,
        'Hispanic (%)': lambda d: d['raceeth2 '] * 100,
        'education 13+ years (%)': lambda d: d['edstg4 13above'] * 100,
        'fair or poor self-rated health (%)': lambda d: (d['shlt'] >= 3).where(d['shlt'].notna()) * 100,
        'any ADL difficulty (%)': lambda d: (d['adl5a'] > 0).where(d['adl5a'].notna()) * 100,
        'number of mobility difficulties (0-5)': lambda d: d['mobila'],
        'ever stroke (%)': lambda d: d['stroke'] * 100,
        'ever diabetes (%)': lambda d: d['diabe'] * 100,
        'hospital stay, previous 2 years (%)': lambda d: d['hosp'] * 100,
    }
    rows = []
    for horizon in HORIZONS:
        dev, ext, P = load_cohorts(horizon)
        for name, cohort in [('2000 baseline', dev), ('2010 baseline', ext)]:
            complete = cohort[P].notna().all(axis=1)
            for label, f in characteristics.items():
                v = f(cohort)
                rows.append({'cohort': name, 'horizon (years)': horizon, 'characteristic': label,
                             'complete predictors': round(v[complete].mean(), 2),
                             'incomplete predictors': round(v[~complete].mean(), 2),
                             'standardized difference': round(_smd(v[~complete], v[complete]), 3),
                             'n complete': int(complete.sum()), 'n incomplete': int((~complete).sum())})
    return pd.DataFrame(rows)


def cohort_overlap():
    rows = []
    for horizon in HORIZONS:
        dev, ext, P = load_cohorts(horizon)
        status = pd.Series(overlap_status(ext['hhidpn'], dev['hhidpn']))
        counts = status.value_counts()
        rows.append({'horizon (years)': horizon, '2010 analytic cohort': len(ext),
                     **{k: int(counts.get(k, 0)) for k in ['returning participant', 'new participant, development household',
                                                           'new participant and household']},
                     'households in 2010 cohort': len(set(household(ext['hhidpn']))),
                     'of which also in 2000 cohort': len(set(household(ext['hhidpn'])) & set(household(dev['hhidpn'])))})
    return pd.DataFrame(rows)


def temporal_by_overlap():
    rows = []
    for horizon in HORIZONS:
        dev, _, _ = load_cohorts(horizon)
        for model in MODELS:
            pred = evaluation.load(horizon, model, "temporal")
            group = overlap_status(pred.hhidpn, dev.hhidpn)
            threshold = evaluation.load(horizon, model, "selection")["temporal"]["threshold"]
            for label, data in pred.groupby(group):
                rows.append({"h": horizon, "model": model, "group": label,
                             "n": len(data), "cases": int(data.y.sum()),
                             "AUROC": 100 * evaluation.discrimination(data.y, data.p, threshold)["AUROC"],
                             "AUPRC": 100 * evaluation.discrimination(data.y, data.p, threshold)["AUPRC"]})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (complete_vs_incomplete, cohort_overlap, temporal_by_overlap):
        atomic_csv(fn(), OUT / f"{fn.__name__}.csv")
