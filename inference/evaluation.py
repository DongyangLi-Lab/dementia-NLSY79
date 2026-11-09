"""Uncertainty, paired comparisons, calibration, net benefit, and runtime from the saved benchmark outputs.

Reads results/h{6,8}/{model}/ (written by benchmark.run) and writes aggregate tables to results/tables/.

- Temporal validation (one shared 2010 test set): household cluster bootstrap (participants of a household are
  resampled together) for each model's metrics and for paired differences between models.
- Internal validation (10 overlapping random splits): mean over repeats with the corrected resampled t interval of
  Nadeau and Bengio (2003), which inflates the between-repeat variance by (1/k + n_test/n_train) to account for the
  overlap between training sets; the same interval is used for paired differences computed within each repeat.
- Calibration: calibration intercept (calibration-in-the-large, slope fixed at 1), calibration slope, observed /
  expected ratio, and Brier score.
- Net benefit (decision curves) over a range of risk thresholds.

Usage: python evaluation.py
"""
import itertools
import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score, precision_score,
                             recall_score, roc_auc_score)

from artifact_store import job_dir
from train.cohorts import ARTIFACTS
from model.specs import MODELS

TABLES = ARTIFACTS / "tables"
OUTCOME = "final"
N_BOOT = 2000
TEST_TRAIN_RATIO = 0.2 / 0.8  # test / (train + validation) in each internal repeat
DCA_THRESHOLDS = np.round(np.arange(0.05, 0.41, 0.01), 2)


def available_models(horizon):
    return [m for m in MODELS if (job_dir("predictions", horizon, OUTCOME, m) / "inference.done.json").is_file()]


def load(horizon, model, kind):
    if kind == "selection":
        path = job_dir("checkpoints", horizon, OUTCOME, model) / "selection.json"
        return json.loads(path.read_text(encoding="utf-8"))
    name = {"metrics": "metrics.csv", "test": "test_predictions.csv", "temporal": "temporal_predictions.csv"}[kind]
    return pd.read_csv(job_dir("predictions", horizon, OUTCOME, model) / name)


def _subset(df, population):
    return df if population == 'all' else df[df['complete'].astype(bool)]


# ---------------------------------------------------------------- metrics

def calibration(y, p):
    """Calibration intercept (slope fixed at 1), calibration slope, observed/expected ratio, Brier score."""
    brier = brier_score_loss(y, p)  # on the unclipped probabilities
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)  # clipping only for the log-odds
    lp = np.log(p / (1 - p))
    y = np.asarray(y, float)
    intercept = sm.GLM(y, np.ones_like(lp), family=sm.families.Binomial(), offset=lp).fit().params[0]
    slope = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1]
    return {'calibration_intercept': intercept, 'calibration_slope': slope, 'OE_ratio': y.mean() / p.mean(),
            'Brier': brier}


def discrimination(y, p, threshold):
    pred = (p >= threshold).astype(int)
    return {'AUROC': roc_auc_score(y, p), 'AUPRC': average_precision_score(y, p),
            'Accuracy': accuracy_score(y, pred), 'Precision': precision_score(y, pred, zero_division=0),
            'Recall': recall_score(y, pred),
            'F1': f1_score(y, pred), 'Brier': brier_score_loss(y, p)}


def calibration_curve(y, p, bins=10):
    """Observed vs mean predicted risk by deciles of predicted risk."""
    d = pd.DataFrame({'y': y, 'p': p})
    d['bin'] = pd.qcut(d['p'], bins, labels=False, duplicates='drop')
    return d.groupby('bin').agg(predicted=('p', 'mean'), observed=('y', 'mean'), n=('y', 'size')).reset_index()


def net_benefit(y, p, thresholds=DCA_THRESHOLDS):
    y = np.asarray(y); p = np.asarray(p); n = len(y); prev = y.mean()
    rows = []
    for t in thresholds:
        pos = p >= t
        tp, fp = np.sum(pos & (y == 1)), np.sum(pos & (y == 0))
        rows.append({'threshold': t, 'net_benefit': tp / n - fp / n * t / (1 - t),
                     'treat_all': prev - (1 - prev) * t / (1 - t), 'treat_none': 0.0})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- uncertainty

def cluster_bootstrap_indices(hhidpn, n_boot=N_BOOT, seed=2026):
    """Row indices of household-cluster bootstrap samples (household = hhidpn // 1000)."""
    hh = np.asarray(hhidpn) // 1000
    codes, uniques = pd.factorize(hh)
    members = [np.flatnonzero(codes == k) for k in range(len(uniques))]
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniques), len(uniques))
        yield np.concatenate([members[k] for k in pick])


def corrected_t_interval(values, ratio=TEST_TRAIN_RATIO, level=0.95):
    """Mean and corrected resampled t interval (Nadeau & Bengio 2003) over k repeats."""
    v = np.asarray(values, float)
    k = len(v)
    se = np.sqrt((1 / k + ratio) * v.var(ddof=1))
    t = stats.t.ppf(0.5 + level / 2, k - 1)
    return v.mean(), v.mean() - t * se, v.mean() + t * se


# ---------------------------------------------------------------- tables

def temporal_table(horizon, population='all', n_boot=N_BOOT):
    """Per-model temporal metrics with household cluster-bootstrap 95% CIs, and paired differences between models."""
    models = available_models(horizon)
    preds = {m: _subset(load(horizon, m, 'temporal'), population).reset_index(drop=True) for m in models}
    thr = {m: load(horizon, m, 'selection')['temporal']['threshold'] for m in models}
    base = preds[models[0]]
    for m in models:
        assert (preds[m]['hhidpn'].to_numpy() == base['hhidpn'].to_numpy()).all(), 'predictions not aligned'
    y = base['y'].to_numpy()
    point = {m: {**discrimination(y, preds[m]['p'].to_numpy(), thr[m]), **calibration(y, preds[m]['p'].to_numpy())}
             for m in models}
    keys = list(point[models[0]])
    boot = {m: {k: [] for k in keys} for m in models}
    for idx in cluster_bootstrap_indices(base['hhidpn'], n_boot):
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        for m in models:
            pb = preds[m]['p'].to_numpy()[idx]
            for k, v in {**discrimination(yb, pb, thr[m]), **calibration(yb, pb)}.items():
                boot[m][k].append(v)
    rows = []
    for m in models:
        for k in keys:
            lo, hi = np.percentile(boot[m][k], [2.5, 97.5])
            rows.append({'horizon': horizon, 'population': population, 'model': m, 'metric': k,
                         'estimate': point[m][k], 'ci_low': lo, 'ci_high': hi, 'n': len(y), 'cases': int(y.sum())})
    pair_rows = []
    for a, b in itertools.combinations(models, 2):
        for k in ['AUROC', 'AUPRC', 'F1', 'Brier']:
            d = np.asarray(boot[a][k]) - np.asarray(boot[b][k])
            est = point[a][k] - point[b][k]
            lo, hi = np.percentile(d, [2.5, 97.5])
            pair_rows.append({'horizon': horizon, 'population': population, 'model_a': a, 'model_b': b, 'metric': k,
                              'difference': est, 'ci_low': lo, 'ci_high': hi,
                              'relative_difference_pct': 100 * est / point[b][k],
                              'p_two_sided': 2 * min(np.mean(d <= 0), np.mean(d >= 0))})
    return pd.DataFrame(rows), pd.DataFrame(pair_rows)


def internal_table(horizon, population='all'):
    """Mean over repeats with corrected resampled t intervals, per model and for paired differences."""
    models = available_models(horizon)
    metrics = {m: load(horizon, m, 'metrics') for m in models}
    per_repeat = {}
    rows = []
    for m in models:
        d = metrics[m][(metrics[m]['setting'] == 'internal') & (metrics[m]['population'] == population)]
        test = _subset(load(horizon, m, 'test'), population)
        cal = pd.DataFrame([calibration(g['y'], g['p']) for _, g in test.groupby('repeat')])
        d = d.sort_values('repeat').reset_index(drop=True)
        d[['calibration_intercept', 'calibration_slope', 'OE_ratio']] = cal[['calibration_intercept',
                                                                               'calibration_slope', 'OE_ratio']]
        per_repeat[m] = d.set_index('repeat')
        for k in ['AUROC', 'AUPRC', 'Accuracy', 'Precision', 'Recall', 'F1', 'Brier', 'calibration_intercept',
                  'calibration_slope', 'OE_ratio']:
            mean, lo, hi = corrected_t_interval(d[k])
            rows.append({'horizon': horizon, 'population': population, 'model': m, 'metric': k, 'estimate': mean,
                         'ci_low': lo, 'ci_high': hi, 'sd': d[k].std(ddof=1), 'repeats': len(d),
                         'n_mean': d['n'].mean(), 'cases_mean': d['cases'].mean()})
    pair_rows = []
    for a, b in itertools.combinations(models, 2):
        for k in ['AUROC', 'AUPRC', 'F1', 'Brier']:
            diff = per_repeat[a][k] - per_repeat[b][k]
            mean, lo, hi = corrected_t_interval(diff)
            se = (hi - mean) / stats.t.ppf(0.975, len(diff) - 1)
            pair_rows.append({'horizon': horizon, 'population': population, 'model_a': a, 'model_b': b, 'metric': k,
                              'difference': mean, 'ci_low': lo, 'ci_high': hi,
                              'relative_difference_pct': 100 * mean / per_repeat[b][k].mean(),
                              'p_two_sided': 2 * stats.t.sf(abs(mean / se), len(diff) - 1) if se > 0 else np.nan})
    return pd.DataFrame(rows), pd.DataFrame(pair_rows)


def calibration_curves(horizon, population='all'):
    rows = []
    for m in available_models(horizon):
        t = _subset(load(horizon, m, 'temporal'), population)
        rows.append(calibration_curve(t['y'], t['p']).assign(horizon=horizon, population=population, model=m,
                                                              setting='temporal'))
        test = _subset(load(horizon, m, 'test'), population)
        rows.append(calibration_curve(test['y'], test['p']).assign(horizon=horizon, population=population, model=m,
                                                                    setting='internal (pooled repeats)'))
    return pd.concat(rows, ignore_index=True)


def decision_curves(horizon, population='all'):
    rows = []
    for m in available_models(horizon):
        t = _subset(load(horizon, m, 'temporal'), population)
        rows.append(net_benefit(t['y'], t['p']).assign(horizon=horizon, population=population, model=m))
    return pd.concat(rows, ignore_index=True)


def horizon_table(metrics=('AUROC', 'AUPRC', 'F1', 'Brier'), n_boot=N_BOOT):
    """8-year minus 6-year performance of each model (all participants).

    The two horizons have different cohorts and prevalences, so differences in AUPRC, F1, and Brier largely reflect
    the higher incidence at 8 years; AUROC is the comparable measure.
    - Internal: the horizons use different cohorts and splits, so repeats are not paired; the difference of the means
      is given a t interval with the two corrected variances added (conservative, as the overlapping cohorts are
      positively correlated).
    - Temporal: joint household cluster bootstrap over the union of the two 2010 cohorts (participants and households
      in both are resampled together).
    """
    models = [m for m in MODELS if m in available_models(6) and m in available_models(8)]
    rows = []
    for m in models:
        per = {}
        for h in (6, 8):
            d = load(h, m, 'metrics')
            per[h] = d[(d['setting'] == 'internal') & (d['population'] == 'all')]
        for k in metrics:
            se2, means = 0, {}
            for h in (6, 8):
                v = per[h][k].to_numpy(float)
                means[h] = v.mean()
                se2 += (1 / len(v) + TEST_TRAIN_RATIO) * v.var(ddof=1)
            diff, se = means[8] - means[6], np.sqrt(se2)
            df = min(len(per[6]), len(per[8])) - 1
            t = stats.t.ppf(0.975, df)
            rows.append({'setting': 'internal', 'model': m, 'metric': k, 'estimate_6y': means[6], 'estimate_8y': means[8],
                         'difference_8y_minus_6y': diff, 'ci_low': diff - t * se, 'ci_high': diff + t * se,
                         'p_two_sided': 2 * stats.t.sf(abs(diff / se), df)})
    preds = {h: {m: load(h, m, 'temporal') for m in models} for h in (6, 8)}
    thr = {h: {m: load(h, m, 'selection')['temporal']['threshold'] for m in models} for h in (6, 8)}
    hh = {h: preds[h][models[0]]['hhidpn'].to_numpy() // 1000 for h in (6, 8)}
    households = np.union1d(hh[6], hh[8])
    members = {}
    for h in (6, 8):
        codes = np.searchsorted(households, hh[h])
        order = np.argsort(codes, kind='stable')
        bounds = np.searchsorted(codes[order], np.arange(len(households) + 1))
        members[h] = [order[bounds[i]:bounds[i + 1]] for i in range(len(households))]

    def score(h, m, idx):
        y, p = preds[h][m]['y'].to_numpy()[idx], preds[h][m]['p'].to_numpy()[idx]
        return {k: v for k, v in discrimination(y, p, thr[h][m]).items() if k in metrics}

    point = {(h, m): score(h, m, np.arange(len(hh[h]))) for h in (6, 8) for m in models}
    boot = {(m, k): [] for m in models for k in metrics}
    rng = np.random.default_rng(2026)
    for _ in range(n_boot):
        pick = rng.integers(0, len(households), len(households))
        idx = {h: np.concatenate([members[h][i] for i in pick]) for h in (6, 8)}
        for m in models:
            s6, s8 = score(6, m, idx[6]), score(8, m, idx[8])
            for k in metrics:
                boot[(m, k)].append(s8[k] - s6[k])
    for m in models:
        for k in metrics:
            d = np.asarray(boot[(m, k)])
            lo, hi = np.percentile(d, [2.5, 97.5])
            rows.append({'setting': 'temporal', 'model': m, 'metric': k, 'estimate_6y': point[(6, m)][k],
                         'estimate_8y': point[(8, m)][k], 'difference_8y_minus_6y': point[(8, m)][k] - point[(6, m)][k],
                         'ci_low': lo, 'ci_high': hi, 'p_two_sided': 2 * min(np.mean(d <= 0), np.mean(d >= 0))})
    return pd.DataFrame(rows)


def runtime_table(horizon):
    rows = []
    for m in available_models(horizon):
        d = job_dir("checkpoints", horizon, OUTCOME, m)
        grid = pd.read_csv(os.path.join(d, 'validation_grid.csv'))
        met = load(horizon, m, 'metrics').drop_duplicates(['setting', 'repeat'])
        internal = met[met["setting"].eq("internal")]
        temporal = met[met["setting"].eq("temporal")]
        device = load(horizon, m, "selection")["device"]
        rows.append({
            "outcome": OUTCOME, "horizon": horizon, "model": m,
            "fits": len(grid) + len(met),
            "validation_fits": len(grid), "internal_refits": len(internal), "temporal_refits": len(temporal),
            "median_seconds_per_fit": float(np.median(np.r_[grid["fit_seconds"], met["fit_seconds"]])),
            "validation_fit_seconds": grid["fit_seconds"].sum(),
            "validation_prediction_seconds": grid["prediction_seconds"].sum(),
            "internal_refit_seconds": internal["fit_seconds"].sum(),
            "internal_prediction_seconds": internal["prediction_seconds"].sum(),
            "temporal_fit_seconds": temporal["fit_seconds"].sum(),
            "temporal_prediction_seconds": temporal["prediction_seconds"].sum(),
            "total_minutes": (grid["fit_seconds"].sum() + met["fit_seconds"].sum()) / 60,
            "total_prediction_minutes": (grid["prediction_seconds"].sum() + met["prediction_seconds"].sum()) / 60,
            "device": device["backend"], "gpu": device.get("gpu_name", ""),
        })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--outcome', default='final', choices=['final', 'any'],
                    help="'any': tables for the outcome-definition sensitivity analysis (results/sensitivity_any)")
    args = ap.parse_args()
    OUTCOME = args.outcome
    prefix = '' if OUTCOME == 'final' else f'sensitivity_{OUTCOME}_'
    for horizon in (6, 8):
        if available_models(horizon) != list(MODELS):
            raise RuntimeError(f'Incomplete {OUTCOME} model outputs for h{horizon}')
    os.makedirs(TABLES, exist_ok=True)
    out = {k: [] for k in ['internal_metrics', 'internal_pairwise', 'temporal_metrics', 'temporal_pairwise',
                           'calibration_curves', 'decision_curves', 'runtime']}
    for horizon in (6, 8):
        for population in ('all', 'complete'):
            a, b = internal_table(horizon, population); out['internal_metrics'].append(a); out['internal_pairwise'].append(b)
            a, b = temporal_table(horizon, population); out['temporal_metrics'].append(a); out['temporal_pairwise'].append(b)
            out['calibration_curves'].append(calibration_curves(horizon, population))
            out['decision_curves'].append(decision_curves(horizon, population))
        out['runtime'].append(runtime_table(horizon))
    if available_models(6) and available_models(8):
        out['horizon_comparison'] = [horizon_table()]
    for name, frames in out.items():
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(os.path.join(TABLES, f'{prefix}{name}.csv'), index=False)
            print('wrote', name)
