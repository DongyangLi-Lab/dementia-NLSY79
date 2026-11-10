"""Figures for the revised manuscript, from the aggregate tables in results/tables/. Writes PDF (vector) and
300-dpi PNG files to figures/.

Fig. 1b participant flow (both baselines and horizons); panel a is the workflow diagram
Fig. 3  internal validation: AUROC, AUPRC, F1, Brier score with 95% CIs, both horizons
Fig. 4  temporal validation (2010 cohort): same layout
Fig. 5  permutation importance on the held-out 2010 cohort (6-year), top 10 per model, with 95% CIs
Fig. 6  calibration in the 2010 cohort (deciles of predicted risk), both horizons
Fig. S1 all participants vs complete cases: AUROC and AUPRC, internal and temporal
Fig. S2 internal validation: precision and recall
Fig. S3 temporal validation: precision and recall
Fig. S4 decision curves (net benefit) in the 2010 cohort, one figure per horizon
Fig. S5 SHAP vs permutation importance ranks (exact SHAP: LR, GNB, DT, LightGBM, CatBoost; KernelSHAP: RealMLP, TabICL, GAT)

Usage: python make_figures.py
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator

ROOT = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(ROOT, 'results', 'tables')
OUT = os.path.join(ROOT, 'figures')

# Model order (6-year internal AUROC) and display names; models are encoded by position, not color.
MODELS = ['TabICL', 'CatBoost', 'LightGBM', 'LR', 'RealMLP', 'GAT', 'DT', 'GNB']
NAMES = {'TabICL': 'TabICL', 'CatBoost': 'CatBoost', 'LightGBM': 'LightGBM', 'LR': 'Penalized LR',
         'RealMLP': 'RealMLP', 'GAT': 'GAT', 'DT': 'Decision tree', 'GNB': 'Gaussian NB'}
# Color encodes the horizon only: categorical slots 1 and 2 of the reference palette.
HORIZON_COLOR = {6: '#2a78d6', 8: '#eb6834'}
HORIZON_LABEL = {6: '6-year horizon', 8: '8-year horizon'}
INK, INK_2, INK_3 = '#0b0b0b', '#52514e', '#8a8984'
GRID = '#e4e3df'
REFERENCE = '#a3a29c'

VARIABLE_LABELS = {
    'ageym': 'Age', 'gender': 'Sex', 'child': 'Living children', 'livpar': 'Living parents',
    'livsib': 'Living siblings', 'momage': "Mother's age", 'dadage': "Father's age", 'edstg': 'Education',
    'lbrf': 'Labor force status', 'mstat': 'Marital status', 'cendiv': 'Census division',
    'raceeth': 'Race/ethnicity', 'shlt': 'Self-rated health', 'slfmem': 'Self-rated memory', 'height': 'Height',
    'weight': 'Weight', 'smokev': 'Ever smoked', 'smoken': 'Current smoker', 'drink': 'Ever drinks alcohol',
    'vigact': 'Vigorous physical activity', 'effort': 'CES-D: everything an effort', 'depres': 'CES-D: felt depressed',
    'sleepr': 'CES-D: restless sleep', 'whappy': 'CES-D: was happy', 'flone': 'CES-D: felt lonely',
    'fsad': 'CES-D: felt sad', 'going': 'CES-D: could not get going', 'enlife': 'CES-D: enjoyed life',
    'hibpe': 'High blood pressure', 'diabe': 'Diabetes', 'cancre': 'Cancer', 'lunge': 'Lung disease',
    'hearte': 'Heart problems', 'stroke': 'Stroke', 'arthre': 'Arthritis', 'hosp': 'Hospital stay (2 y)',
    'nrshom': 'Any nursing-home stay (2 y)', 'nrstim': 'Nursing-home stays (2 y)',
    'nrsnit': 'Nursing-home nights (2 y)', 'doctim': 'Doctor visits (2 y)', 'hlthlm': 'Health limits work',
    'proxy': 'Proxy interview', 'toilta': 'Difficulty using toilet', 'adl5a': 'ADL difficulties (0-5)',
    'mapa': 'Difficulty using a map', 'walksa': 'Walking several blocks', 'walk1a': 'Walking one block',
    'sita': 'Sitting 2 hours', 'chaira': 'Getting up from a chair', 'climsa': 'Climbing several flights',
    'clim1a': 'Climbing one flight', 'stoopa': 'Stooping/kneeling', 'lifta': 'Lifting 10 lb',
    'dimea': 'Picking up a dime', 'armsa': 'Reaching overhead', 'pusha': 'Pushing/pulling objects',
    'mobila': 'Mobility difficulties (0-5)', 'lgmusa': 'Large-muscle difficulties (0-4)',
    'grossa': 'Gross-motor difficulties (0-5)', 'finea': 'Fine-motor difficulties (0-3)',
}

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5, 'axes.edgecolor': INK_3,
    'axes.labelcolor': INK_2, 'xtick.color': INK_2, 'ytick.color': INK_2, 'text.color': INK,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'pdf.fonttype': 42,
    'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
})


def _style(ax, grid_axis='x'):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'), dpi=300, bbox_inches='tight',
                    metadata={'CreationDate': None} if ext == 'pdf' else None)  # reproducible files
    plt.close(fig)
    print('wrote', name)


def _horizon_legend(fig, y=1.0):
    handles = [plt.Line2D([], [], marker='o', color=HORIZON_COLOR[h], linewidth=1.4, markersize=4.5,
                          label=HORIZON_LABEL[h]) for h in (6, 8)]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, y), ncol=2, frameon=False)


# ------------------------------------------------------------------ Figs. 3 and 4: performance with CIs

def performance_figure(table, name, title, metrics=None):
    df = pd.read_csv(os.path.join(TABLES, table))
    df = df[df['population'] == 'all']
    metrics = metrics or [('AUROC', 'AUROC', 100), ('AUPRC', 'AUPRC', 100), ('F1', 'F1 score', 100),
                          ('Brier', 'Brier score', 1)]
    fig, axes = plt.subplots(1, len(metrics), figsize=(1.8 * len(metrics) + 0.6, 3.0), sharey=True)
    y = np.arange(len(MODELS))[::-1]
    offset = {6: 0.17, 8: -0.17}
    for ax, (key, label, scale) in zip(axes, metrics):
        for h in (6, 8):
            d = df[(df['horizon'] == h) & (df['metric'] == key)].set_index('model').loc[MODELS]
            est, lo, hi = d['estimate'] * scale, d['ci_low'] * scale, d['ci_high'] * scale
            ax.hlines(y + offset[h], lo, hi, color=HORIZON_COLOR[h], linewidth=1.4)
            ax.plot(est, y + offset[h], 'o', color=HORIZON_COLOR[h], markersize=4.5,
                    markeredgecolor='white', markeredgewidth=0.6)
        ax.set_title(label + (' (%)' if scale == 100 else ' (lower is better)'), color=INK)
        _style(ax)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([NAMES[m] for m in MODELS])
    for ax in axes:
        ax.tick_params(axis='y', length=0)
    fig.suptitle(title, x=0.01, ha='left', y=1.08, fontsize=9, color=INK)
    _horizon_legend(fig, y=1.02)
    fig.tight_layout()
    _save(fig, name)


# ------------------------------------------------------------------ Fig. S1: all participants vs complete cases

def population_figure():
    """Primary evaluation (all participants, training-only imputation) vs complete cases. Horizon is the color;
    population is the marker fill (filled = all, open = complete cases)."""
    panels = [('internal_metrics.csv', 'AUROC', 'Internal: AUROC (%)'), ('internal_metrics.csv', 'AUPRC', 'Internal: AUPRC (%)'),
              ('temporal_metrics.csv', 'AUROC', 'Temporal: AUROC (%)'), ('temporal_metrics.csv', 'AUPRC', 'Temporal: AUPRC (%)')]
    fig, axes = plt.subplots(2, 4, figsize=(7.4, 5.4), sharey=True)
    y = np.arange(len(MODELS))[::-1]
    offset = {'all': 0.17, 'complete': -0.17}
    for r, h in enumerate((6, 8)):
        for c, (table, key, label) in enumerate(panels):
            ax = axes[r, c]
            df = pd.read_csv(os.path.join(TABLES, table))
            for pop in ('all', 'complete'):
                d = df[(df['horizon'] == h) & (df['metric'] == key) & (df['population'] == pop)].set_index('model').loc[MODELS]
                ax.hlines(y + offset[pop], d['ci_low'] * 100, d['ci_high'] * 100, color=HORIZON_COLOR[h], linewidth=1.2)
                ax.plot(d['estimate'] * 100, y + offset[pop], 'o', markersize=4.5, markeredgewidth=1.0,
                        color=HORIZON_COLOR[h], markerfacecolor=HORIZON_COLOR[h] if pop == 'all' else 'white')
            ax.set_title(f'{HORIZON_LABEL[h]}\n{label}', color=INK)
            ax.tick_params(axis='y', length=0)
            _style(ax)
        axes[r, 0].set_yticks(y)
        axes[r, 0].set_yticklabels([NAMES[m] for m in MODELS])
    handles = [plt.Line2D([], [], marker='o', color=INK_2, linewidth=1.2, markersize=4.5, markerfacecolor=f, label=l)
               for f, l in ((INK_2, 'All participants (primary)'), ('white', 'Complete cases'))]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    fig.suptitle('Evaluation population: all participants vs complete cases (95% CIs)', x=0.01, ha='left', y=1.06,
                 fontsize=9, color=INK)
    fig.tight_layout()
    _save(fig, 'figS1_all_vs_complete_cases')


# ------------------------------------------------------------------ Fig. 5: permutation importance

def importance_figure():
    """Age dominates every model (about 20 points), so it is reported in each panel's subtitle and the bars show
    the next 10 predictors on their own scale."""
    df = pd.read_csv(os.path.join(TABLES, 'importance_permutation_h6.csv'))
    fig, axes = plt.subplots(2, 4, figsize=(7.4, 5.6))
    for ax, m in zip(axes.ravel(), MODELS):
        d = df[df['model'] == m].sort_values('importance_AUPRC_drop', ascending=False)
        age = d[d['variable'] == 'ageym'].iloc[0]
        d = d[d['variable'] != 'ageym'].head(10)[::-1]
        yy = np.arange(len(d))
        imp, lo, hi = d['importance_AUPRC_drop'] * 100, d['ci_low'] * 100, d['ci_high'] * 100
        ax.barh(yy, imp, height=0.62, color=HORIZON_COLOR[6], edgecolor='white', linewidth=0.5)
        ax.hlines(yy, lo, hi, color=INK_2, linewidth=0.8)
        ax.set_yticks(yy)
        ax.set_yticklabels([VARIABLE_LABELS[v] for v in d['variable']], fontsize=6.6)
        ax.tick_params(axis='y', length=0)
        ax.set_title(f"{NAMES[m]}\nAge (rank {int(age['rank'])}): {age['importance_AUPRC_drop'] * 100:.1f} "
                     f"({age['ci_low'] * 100:.1f}-{age['ci_high'] * 100:.1f})", color=INK, fontsize=7.6)
        ax.set_xlim(left=min(0, lo.min() * 1.1))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        _style(ax)
    fig.supxlabel('Decrease in AUPRC when the predictor is permuted (percentage points; 95% CI)', fontsize=8,
                  color=INK_2)
    fig.suptitle('Permutation importance on the held-out 2010 cohort (6-year horizon): age, then the next 10 '
                 'predictors per model', x=0.01, ha='left', fontsize=9, color=INK)
    fig.tight_layout()
    _save(fig, 'fig5_feature_importance')


# ------------------------------------------------------------------ Fig. 6: calibration

def calibration_figure():
    df = pd.read_csv(os.path.join(TABLES, 'calibration_curves.csv'))
    df = df[(df['population'] == 'all') & (df['setting'] == 'temporal')]
    metrics = pd.read_csv(os.path.join(TABLES, 'temporal_metrics.csv'))
    metrics = metrics[metrics['population'] == 'all']
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.0), sharex=True, sharey=True)
    for ax, m in zip(axes.ravel(), MODELS):
        ax.plot([0, 1], [0, 1], color=REFERENCE, linewidth=0.8)
        notes = []
        for h in (6, 8):
            d = df[(df['model'] == m) & (df['horizon'] == h)]
            ax.plot(d['predicted'], d['observed'], '-o', color=HORIZON_COLOR[h], linewidth=1.2, markersize=3.4,
                    markeredgecolor='white', markeredgewidth=0.5)
            oe = metrics[(metrics['model'] == m) & (metrics['horizon'] == h) & (metrics['metric'] == 'OE_ratio')]
            sl = metrics[(metrics['model'] == m) & (metrics['horizon'] == h) & (metrics['metric'] == 'calibration_slope')]
            notes.append(f'{h}-y: O/E {oe["estimate"].iloc[0]:.2f}, slope {sl["estimate"].iloc[0]:.2f}')
        ax.text(0.03, 0.97, '\n'.join(notes), transform=ax.transAxes, ha='left', va='top', fontsize=6.2,
                color=INK_2)
        ax.set_title(NAMES[m], color=INK)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        _style(ax, grid_axis='both')
    for ax in axes[1]:
        ax.set_xlabel('Mean predicted risk')
    for ax in axes[:, 0]:
        ax.set_ylabel('Observed incidence')
    fig.suptitle('Calibration in the 2010 temporal-validation cohort (deciles of predicted risk)', x=0.01, ha='left',
                 y=1.06, fontsize=9, color=INK)
    _horizon_legend(fig, y=1.01)
    fig.tight_layout()
    _save(fig, 'fig6_calibration')


# ------------------------------------------------------------------ Fig. S4: decision curves

def decision_curve_figure(horizon):
    df = pd.read_csv(os.path.join(TABLES, 'decision_curves.csv'))
    df = df[(df['population'] == 'all') & (df['horizon'] == horizon)]
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.8), sharex=True, sharey=True)
    for ax, m in zip(axes.ravel(), MODELS):
        d = df[df['model'] == m].sort_values('threshold')
        ax.axhline(0, color=INK, linewidth=0.7)
        ax.plot(d['threshold'] * 100, d['treat_all'], color=REFERENCE, linewidth=1.0)
        ax.plot(d['threshold'] * 100, d['net_benefit'], color=HORIZON_COLOR[horizon], linewidth=1.6)
        ax.set_title(NAMES[m], color=INK)
        ax.set_ylim(-0.03, 0.14)
        _style(ax, grid_axis='y')
    for ax in axes[1]:
        ax.set_xlabel('Risk threshold (%)')
    for ax in axes[:, 0]:
        ax.set_ylabel('Net benefit')
    handles = [plt.Line2D([], [], color=HORIZON_COLOR[horizon], linewidth=1.6, label='Model'),
               plt.Line2D([], [], color=REFERENCE, linewidth=1.0, label='Treat all'),
               plt.Line2D([], [], color=INK, linewidth=0.7, label='Treat none')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False)
    fig.suptitle(f'Decision curves in the 2010 temporal-validation cohort ({horizon}-year horizon)', x=0.01, ha='left',
                 y=1.06, fontsize=9, color=INK)
    fig.tight_layout()
    _save(fig, f'figS4_decision_curves_{horizon}y')


# ------------------------------------------------------------------ Fig. S5: SHAP vs permutation ranks

def shap_figure():
    perm = pd.read_csv(os.path.join(TABLES, 'importance_permutation_h6.csv'))
    shap = pd.read_csv(os.path.join(TABLES, 'importance_shap_h6.csv'))
    agree = pd.read_csv(os.path.join(TABLES, 'importance_rank_agreement_h6.csv')).set_index('model')
    models = [m for m in MODELS if m in agree.index]
    method = agree['method'].fillna('exact') if 'method' in agree else pd.Series('exact', index=agree.index)
    approx = set(method.index[method != 'exact'])
    ncol = 3
    nrow = -(-(len(models) + 1) // ncol)  # one spare panel for the notes
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.2, 2.9 * nrow), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax in axes[len(models):]:
        ax.axis('off')
    notes = []
    for ax, m in zip(axes, models):
        d = perm[perm['model'] == m][['variable', 'rank']].merge(
            shap[shap['model'] == m][['variable', 'rank']], on='variable', suffixes=('_perm', '_shap'))
        ax.plot([1, 60], [1, 60], color=REFERENCE, linewidth=0.8)
        if m in approx:  # open markers: approximate (KernelSHAP) values
            ax.plot(d['rank_perm'], d['rank_shap'], 'o', color=HORIZON_COLOR[6], markersize=3.6,
                    markerfacecolor='white', markeredgewidth=0.8)
        else:
            ax.plot(d['rank_perm'], d['rank_shap'], 'o', color=HORIZON_COLOR[6], markersize=3.6, alpha=0.85,
                    markeredgecolor='white', markeredgewidth=0.4)
        top3 = d.sort_values('rank_perm').head(3)['variable']
        notes.append(f'{NAMES[m]}:\n' + ', '.join(VARIABLE_LABELS[v] for v in top3))
        unused = int((shap[shap['model'] == m]['mean_abs_SHAP'] == 0).sum())
        if unused > 10:  # predictors never used by the model share one tied rank (one point)
            ax.text(0.97, 0.97, f'{unused} predictors not used\nby the model are tied\n(one point)',
                    transform=ax.transAxes, ha='right', va='top', fontsize=6.4, color=INK_2)
        a = agree.loc[m]
        ax.set_title(f"{NAMES[m]}{' (approximate)' if m in approx else ''}\nSpearman {a['spearman_rank_correlation']:.2f}; "
                     f"top 10 shared: {int(a['shared_top10'])}", color=INK)
        ax.set_xlim(0, 61)
        ax.set_ylim(61, 0)
        _style(ax, grid_axis='both')
        ax.tick_params(labelbottom=True)
    if len(models) < len(axes):
        # with all eight models the top-3 lists do not fit; the top predictors are shown in Fig. 5
        note = ('Filled: exact SHAP\nOpen: approximate SHAP\n(KernelSHAP, 500 participants)' if approx else
                'Top 3 predictors by permutation importance\n' + '\n'.join(notes))
        axes[len(models)].text(0.02, 0.98, note,
                               transform=axes[len(models)].transAxes, ha='left', va='top', fontsize=6.6,
                               color=INK_2, linespacing=1.5)
    fig.supxlabel('Permutation-importance rank', fontsize=8, color=INK_2)
    fig.supylabel('SHAP rank', fontsize=8, color=INK_2)
    fig.suptitle('Agreement between SHAP and permutation importance ranks (6-year horizon, 2010 cohort)', x=0.01,
                 ha='left', fontsize=9, color=INK)
    fig.tight_layout()
    _save(fig, 'figS5_shap_vs_permutation')


# ------------------------------------------------------------------ Fig. 1: participant flow

def flow_figure():
    flow = pd.read_csv(os.path.join(TABLES, 'participant_flow.csv'))
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    def box(x, y, w, h, text, fill='#f3f2ef', edge=INK_3, size=7.2, weight='normal'):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.2', facecolor=fill,
                                    edgecolor=edge, linewidth=0.7))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=size, color=INK, weight=weight,
                linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='-|>', color=INK_3, linewidth=0.8, shrinkA=0, shrinkB=0))

    for col, (year, x) in enumerate([(2000, 2), (2010, 52)]):
        f6 = flow[(flow['cohort'] == f'{year} baseline') & (flow['horizon (years)'] == 6)].iloc[0]
        f8 = flow[(flow['cohort'] == f'{year} baseline') & (flow['horizon (years)'] == 8)].iloc[0]
        role = 'Development cohort' if year == 2000 else 'Temporal-validation cohort'
        n = lambda v: f'{int(v):,}'
        box(x, 88, 46, 8, f'{role}\nHRS {year}: interviewed, n = {n(f6["interviewed at baseline"])}',
            fill='#e8f0fb', weight='bold')
        box(x + 4, 70, 38, 13,
            f'Excluded\nAged 50 or younger: {n(f6["excluded: age <= 50"])}\n'
            f'Dementia at baseline: {n(f6["excluded: dementia at baseline"])}\n'
            f'Baseline dementia status unknown: {n(f6["excluded: baseline dementia status unknown"])}', size=6.8)
        box(x, 58, 46, 7, f'Eligible: aged > 50, dementia-free at baseline\nn = {n(f6["eligible"])}')
        arrow(x + 23, 88, x + 23, 83)
        arrow(x + 23, 70, x + 23, 65)
        for row, (f, h) in enumerate([(f6, 6), (f8, 8)]):
            bx = x + row * 23.5
            box(bx, 33, 22.5, 19,
                f'{h}-year horizon\nExcluded:\ndied before final wave,\nno dementia: {n(f["excluded: died before final wave, no dementia observed"])}\n'
                f'not assessed at\nfinal wave: {n(f["excluded: alive or unknown, not assessed at final wave"])}', size=6.3)
            box(bx, 13, 22.5, 14,
                f'Analytic cohort\nn = {n(f["analytic cohort"])}\nincident dementia: {n(f["incident dementia"])}\n'
                f'complete predictors:\n{n(f["complete predictors"])} ({n(f["incident dementia, complete predictors"])})',
                fill='#fdf0e9' if h == 8 else '#e8f0fb', size=6.3)
            arrow(x + 23, 58, bx + 11.25, 52)
            arrow(bx + 11.25, 33, bx + 11.25, 27)
    ax.text(50, 4, 'Development: random stratified 60/20/20 splits, 10 repeats; temporal validation: models fitted on the '
                   'full 2000 cohort and evaluated once on the 2010 cohort.\nParentheses: incident dementia among '
                   'participants with complete predictors (complete-case sensitivity analysis).',
            ha='center', va='center', fontsize=6.6, color=INK_2)
    _save(fig, 'fig1_participant_flow')


if __name__ == '__main__':
    flow_figure()
    performance_figure('internal_metrics.csv', 'fig3_internal_validation',
                       'Internal validation (2000 cohort; mean over 10 repeats, corrected 95% CIs)')
    performance_figure('temporal_metrics.csv', 'fig4_temporal_validation',
                       'Temporal validation within the HRS (2010 cohort; household-bootstrap 95% CIs)')
    pr = [('Precision', 'Precision', 100), ('Recall', 'Recall', 100)]
    performance_figure('internal_metrics.csv', 'figS2_internal_precision_recall',
                       'Internal validation: precision and recall at the F1-maximizing threshold', pr)
    performance_figure('temporal_metrics.csv', 'figS3_temporal_precision_recall',
                       'Temporal validation: precision and recall at the development threshold', pr)
    population_figure()
    importance_figure()
    calibration_figure()
    for h in (6, 8):
        decision_curve_figure(h)
    shap_figure()
