"""Tabular sources for the submitted manuscript and supplement numbering."""
import json
import os
from collections import Counter

import numpy as np
import pandas as pd

from make_figures import MODELS, NAMES, VARIABLE_LABELS

from inference import evaluation
from train.cohorts import ARTIFACTS

TABLES = ARTIFACTS / "tables"
OUT = os.path.join(TABLES, 'supplementary')
COHORT = {'temporal': '2010 cohort', 'internal': '2000 cohort'}
PCT = ('AUROC', 'AUPRC', 'Accuracy', 'Precision', 'Recall', 'F1')
MAIN_METRICS = ['AUROC', 'AUPRC', 'Precision', 'Recall', 'F1', 'Brier', 'calibration_intercept',
                'calibration_slope', 'OE_ratio']
METRIC_LABEL = {'AUROC': 'AUROC (%)', 'AUPRC': 'AUPRC (%)', 'Accuracy': 'Accuracy (%)', 'Precision': 'Precision (%)',
                'Recall': 'Recall (%)', 'F1': 'F1 (%)', 'Brier': 'Brier score', 'calibration_intercept':
                'Calibration intercept', 'calibration_slope': 'Calibration slope', 'OE_ratio': 'O/E ratio'}


def read(name):
    return pd.read_csv(os.path.join(TABLES, name))


def fmt(est, lo=None, hi=None, metric='AUROC'):
    if metric in PCT:
        s, f = 100, '{:.1f}'
    elif metric == 'Brier':
        s, f = 1, '{:.3f}'
    else:
        s, f = 1, '{:.2f}'
    out = f.format(est * s)
    if lo is not None:
        out += f' ({f.format(lo * s)} to {f.format(hi * s)})'
    return out


def ordered(df, col='model'):
    return df.assign(_o=df[col].map({m: i for i, m in enumerate(MODELS)})).sort_values(
        [c for c in ('horizon', 'h') if c in df] + ['_o']).drop(columns='_o')


def metric_table(df, metrics, population='all'):
    """Wide table: one row per horizon and model, one column per metric, 'estimate (95% CI)'."""
    df = ordered(df[df['population'] == population])
    rows = []
    for (h, m), g in df.groupby(['horizon', 'model'], sort=False):
        g = g.set_index('metric')
        n = g['n'].iloc[0] if 'n' in g else g['n_mean'].iloc[0]
        cases = g['cases'].iloc[0] if 'cases' in g else g['cases_mean'].iloc[0]
        row = {'Horizon': f'{h}-year', 'Model': NAMES[m], 'n': f'{n:,.0f}', 'Cases': f'{cases:,.0f}',
               'Prevalence (%)': f'{100 * cases / n:.1f}'}
        for k in metrics:
            row[METRIC_LABEL[k]] = fmt(g.at[k, 'estimate'], g.at[k, 'ci_low'], g.at[k, 'ci_high'], k)
        rows.append(row)
    return pd.DataFrame(rows)


def s1_outcome_status():
    return read("S1_outcome_flow.csv")


def t11_complete_vs_incomplete():
    d = read('complete_vs_incomplete.csv')
    return pd.DataFrame({
        'Cohort': d['cohort'], 'Horizon': d['horizon (years)'].map(lambda h: f'{h}-year'),
        'Characteristic': d['characteristic'],
        'Complete predictors': [f'{a:.2f}' if max(a, b) < 10 else f'{a:.1f}' for a, b in zip(d['complete predictors'], d['incomplete predictors'])],
        'Incomplete predictors': [f'{b:.2f}' if max(a, b) < 10 else f'{b:.1f}' for a, b in zip(d['complete predictors'], d['incomplete predictors'])],
        'Standardized difference': d['standardized difference'].map(lambda v: f'{v:.2f}'),
        'n complete / incomplete': [f'{a:,} / {b:,}' for a, b in zip(d['n complete'], d['n incomplete'])]})


def t12_overlap():
    o = read('cohort_overlap.csv')
    b = ordered(read('temporal_by_overlap.csv'))
    groups = ['returning participant', 'new participant, development household', 'new participant and household']
    rows = []
    for _, r in o.iterrows():
        h, n = r['horizon (years)'], r['2010 analytic cohort']
        for g in groups:
            first = b[(b['h'] == h) & (b['group'] == g)].iloc[0]
            row = {'Horizon': f'{h}-year', 'Group': g[0].upper() + g[1:],
                   'n (% of cohort)': f'{r[g]:,} ({100 * r[g] / n:.1f})',
                   'Cases (%)': f'{first["cases"]:,} ({100 * first["cases"] / first["n"]:.1f})'}
            for _, m in b[(b['h'] == h) & (b['group'] == g)].iterrows():
                row[f'{NAMES[m["model"]]} AUROC / AUPRC (%)'] = f'{m["AUROC"]:.1f} / {m["AUPRC"]:.1f}'
            rows.append(row)
        rows.append({'Horizon': f'{h}-year', 'Group': 'Households in 2010 cohort, of which also in 2000 cohort',
                     'n (% of cohort)': f'{r["households in 2010 cohort"]:,}, of which {r["of which also in 2000 cohort"]:,} '
                                        f'({100 * r["of which also in 2000 cohort"] / r["households in 2010 cohort"]:.1f})'})
    return pd.DataFrame(rows).fillna('')


def t13_accuracy_complete():
    i, t = read('internal_metrics.csv'), read('temporal_metrics.csv')
    metrics = ['AUROC', 'AUPRC', 'Accuracy', 'Precision', 'Recall', 'F1', 'Brier', 'calibration_slope', 'OE_ratio']
    acc, complete = [], []
    for setting, d in (('Internal (2000 cohort)', i), ('Temporal (2010 cohort)', t)):
        a = metric_table(d, ['Accuracy'])
        a.insert(0, 'Validation', setting)
        acc.append(a)
        c = metric_table(d, metrics, population='complete')
        c.insert(0, 'Validation', setting)
        complete.append(c)
    return pd.concat(acc, ignore_index=True), pd.concat(complete, ignore_index=True)


def pairwise_table(name):
    d = read(name)
    d = d[d['population'] == 'all']
    rows = []
    for _, r in d.iterrows():
        a, b = r['model_a'], r['model_b']
        # orient each pair so that model A comes first in the fixed model order (higher 6-year internal AUROC)
        base_b = r['difference'] / (r['relative_difference_pct'] / 100)  # comparator (model_b) value
        diff, lo, hi = r['difference'], r['ci_low'], r['ci_high']
        if MODELS.index(a) > MODELS.index(b):
            a, b = b, a
            base_b = base_b + diff  # the new comparator is the former model_a
            diff, lo, hi = -diff, -hi, -lo
        p = r['p_two_sided']
        rows.append({'horizon': r['horizon'], 'oa': MODELS.index(a), 'ob': MODELS.index(b), 'metric': r['metric'],
                     'Horizon': f'{r["horizon"]}-year', 'Model A': NAMES[a], 'Model B': NAMES[b],
                     'Metric': METRIC_LABEL[r['metric']].replace(' (%)', ''),
                     'Absolute difference, A - B (95% CI)': (
                         f'{100 * diff:.1f} ({100 * lo:.1f} to {100 * hi:.1f}) points' if r['metric'] != 'Brier'
                         else f'{diff:.4f} ({lo:.4f} to {hi:.4f})'),
                     'Relative difference (%)': f'{100 * diff / base_b:.1f}',
                     'p': '<0.001' if p < 0.001 else f'{p:.3f}'})
    order = {'AUROC': 0, 'AUPRC': 1, 'F1': 2, 'Brier': 3}
    out = pd.DataFrame(rows)
    out = out.assign(om=out['metric'].map(order)).sort_values(['horizon', 'om', 'oa', 'ob'])
    return out.drop(columns=['horizon', 'oa', 'ob', 'metric', 'om'])


def horizon_comparison(setting):
    d = read('horizon_comparison.csv')
    d = ordered(d[d['setting'] == setting])
    order = {'AUROC': 0, 'AUPRC': 1, 'F1': 2, 'Brier': 3}
    d = d.assign(om=d['metric'].map(order)).sort_values('om', kind='stable')
    rows = []
    for _, r in d.iterrows():
        k, p = r['metric'], r['p_two_sided']
        diff = (f'{100 * r["difference_8y_minus_6y"]:.1f} ({100 * r["ci_low"]:.1f} to {100 * r["ci_high"]:.1f}) points'
                if k != 'Brier' else f'{r["difference_8y_minus_6y"]:.4f} ({r["ci_low"]:.4f} to {r["ci_high"]:.4f})')
        rows.append({'Metric': METRIC_LABEL[k], 'Model': NAMES[r['model']], '6-year': fmt(r['estimate_6y'], metric=k),
                     '8-year': fmt(r['estimate_8y'], metric=k), 'Difference, 8-year - 6-year (95% CI)': diff,
                     'p': '<0.001' if p < 0.001 else f'{p:.3f}'})
    return pd.DataFrame(rows)


def _params(p):
    return ', '.join(f'{k}={v}' for k, v in p.items()) if p else '(defaults)'


def t16_selection():
    rows = []
    for h in (6, 8):
        for m in MODELS:
            s = evaluation.load(h, m, 'selection')
            reps = s['per_repeat']
            counts = Counter(_params(r['params']) for r in reps)
            thr = np.array([r['threshold'] for r in reps])
            rows.append({'Horizon': f'{h}-year', 'Model': NAMES[m],
                         'Configurations selected in internal repeats (times of 10)':
                             '; '.join(f'{c} [{n}]' for c, n in counts.most_common()),
                         'Internal thresholds, median (range)': f'{np.median(thr):.3f} ({thr.min():.3f} to {thr.max():.3f})',
                         'Temporal-validation configuration': _params(s['temporal']['params']),
                         'Temporal-validation threshold': f'{s["temporal"]["threshold"]:.3f}'})
    return pd.DataFrame(rows)


def t17_runtime():
    d = ordered(read('runtime.csv'))
    return pd.DataFrame({'Horizon': d['horizon'].map(lambda h: f'{h}-year'), 'Model': d['model'].map(NAMES),
                         'Fits': d['fits'], 'Median seconds per fit': d['median_seconds_per_fit'].map(lambda v: f'{v:.2f}'),
                         'Total minutes': d['total_minutes'].map(lambda v: f'{v:.1f}')})


def t18_shap():
    perm, shap = read('importance_permutation_h6.csv'), read('importance_shap_h6.csv')
    agree = ordered(read('importance_rank_agreement_h6.csv'))
    method = agree['method'].fillna('exact') if 'method' in agree else pd.Series('exact', index=agree.index)
    a = pd.DataFrame({'Model': agree['model'].map(NAMES),
                      'SHAP method': [('KernelSHAP (approximate; 500 participants)' if k != 'exact' else
                                       {'LR': 'exact (linear)', 'GNB': 'exact (additive log-odds)'}.get(m, 'exact (tree SHAP)'))
                                      for m, k in zip(agree['model'], method)],
                      'SHAP scale': ['probability' if m == 'DT' else 'log-odds' for m in agree['model']],
                      'Spearman rank correlation (60 predictors)': agree['spearman_rank_correlation'].map(lambda v: f'{v:.2f}'),
                      'Shared top-10 predictors': agree['shared_top10']})
    rows = []
    for m in agree['model']:
        p = perm[perm['model'] == m].set_index('variable')
        s = shap[shap['model'] == m].set_index('variable')
        top = sorted(set(p.index[p['rank'] <= 10]) | set(s.index[s['rank'] <= 10]), key=lambda v: (p.at[v, 'rank'], s.at[v, 'rank']))
        for v in top:
            rows.append({'Model': NAMES[m], 'Predictor': VARIABLE_LABELS.get(v, v),
                         'Permutation rank': p.at[v, 'rank'],
                         'Permutation importance, AUPRC drop (95% CI), points': fmt(p.at[v, 'importance_AUPRC_drop'],
                                                                                   p.at[v, 'ci_low'], p.at[v, 'ci_high'], 'AUPRC'),
                         'SHAP rank': s.at[v, 'rank'], 'Mean |SHAP|': f'{s.at[v, "mean_abs_SHAP"]:.3f}'})
    return a, pd.DataFrame(rows)


def t18c_kernel_check():
    d = ordered(read('importance_kernel_shap_check_h6.csv'))
    return pd.DataFrame({'Model': d['model'].map(NAMES),
                         'Spearman rank correlation, approximate vs exact SHAP': d['spearman_rank_correlation'].map(lambda v: f'{v:.2f}'),
                         'Shared top-10 predictors': d['shared_top10']})


def t19_sensitivity():
    out = []
    metrics = ['AUROC', 'AUPRC', 'F1', 'Brier']
    for setting, prim, sens in (('Internal (2000 cohort)', 'internal_metrics.csv', 'sensitivity_any_internal_metrics.csv'),
                                ('Temporal (2010 cohort)', 'temporal_metrics.csv', 'sensitivity_any_temporal_metrics.csv')):
        s = read(sens)
        p = read(prim)
        p = p[p['model'].isin(s['model'].unique())]
        for label, d in (('Assessed dementia-free at any follow-up wave (sensitivity)', s),
                         ('Assessed dementia-free at the final wave (primary)', p)):
            w = metric_table(d, metrics)
            w.insert(0, 'Non-case definition', label)
            w.insert(0, 'Validation', setting)
            out.append(w)
    out = pd.concat(out, ignore_index=True)
    return out.sort_values(['Validation', 'Horizon', 'Model'], key=lambda c: c.map(
        {NAMES[m]: i for i, m in enumerate(MODELS)}) if c.name == 'Model' else c, kind='stable')


def build():
    i, t = read('internal_metrics.csv'), read('temporal_metrics.csv')
    s13a, s13b = t13_accuracy_complete()
    s18a, s18b = t18_shap()
    s18c = t18c_kernel_check()
    main = pd.concat([metric_table(i, MAIN_METRICS).assign(Validation="Internal"),
                      metric_table(t, MAIN_METRICS).assign(Validation="Temporal")], ignore_index=True)
    main = main[["Validation"] + [c for c in main if c != "Validation"]]
    internal = metric_table(i, MAIN_METRICS)
    temporal = metric_table(t, MAIN_METRICS)
    tables = {
        "Table1": main,
        "S1": s1_outcome_status(),
        "S11": t11_complete_vs_incomplete(),
        "S13": s13b,
        "S14": t12_overlap(),
        "S15": t16_selection(),
        "S16": pairwise_table("internal_pairwise.csv"),
        "S17": pairwise_table("temporal_pairwise.csv"),
        "S18": horizon_comparison("internal"),
        "S19": horizon_comparison("temporal"),
        "S20": t17_runtime(),
        "S21": s18c,
        "S22a": internal[internal.Horizon.eq("6-year")],
        "S22b": internal[internal.Horizon.eq("8-year")],
        "S23": s13a,
        "S24a": temporal[temporal.Horizon.eq("6-year")],
        "S24b": temporal[temporal.Horizon.eq("8-year")],
        "S25": s18b,
        "S26": s18a,
        "S27": t19_sensitivity(),
    }
    for name, file in (("S9", "S9_predictor_summary.csv"), ("S10", "S10_predictor_distribution.csv"),
                       ("S12", "S12_imputation_comparison_h6.csv")):
        if (TABLES / file).exists():
            tables[name] = read(file)
    return tables


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name, frame in build().items():
        frame.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
        print(f"wrote {name}")
