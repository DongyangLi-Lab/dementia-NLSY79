"""Validate revised results and derive plotting tables without fitting models."""
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd

MODELS = ['LR', 'GNB', 'DT', 'LightGBM', 'CatBoost', 'RealMLP', 'GAT', 'TabICL']


def prepare(result, out):
    result, out = Path(result), Path(out)
    tables = result / 'tables/audited_sources'
    prep = out / 'input_tables'
    for folder in ['source_data', 'qa', 'preview', 'cache', 'captions', 'selected_component', 'tables', 'supplement']:
        (out / folder).mkdir(parents=True, exist_ok=True)
    for folder in ['fig1_observation', 'fig2_coverage', 'fig4_signal', 'fig5_reliability']:
        (prep / folder).mkdir(parents=True, exist_ok=True)
    flow = pd.read_csv(tables / 'participant_flow.csv')
    checks, sequences, missing = [], [], []
    for year in (2000, 2010):
        for h in (6, 8):
            window = f'{year}-{year+h}'
            d = pd.read_csv(result / 'cohorts' / f'{window}_final.csv')
            audit = pd.read_csv(result / 'cohorts' / f'{window}_audit.csv').set_index('hhidpn')
            row = flow[(flow.cohort == f'{year} baseline') & (flow['horizon (years)'] == h)].iloc[0]
            assert d.hhidpn.is_unique and d.y.isin([0, 1]).all()
            assert len(d) == row['analytic cohort'] and d.y.sum() == row['incident dementia']
            a = audit.loc[d.hhidpn]
            assert a.alive_at_final_wave.all() and not a.death_by_final_wave.any()
            assert np.array_equal(a.y_final.to_numpy(), d.y.to_numpy())
            # Sensitivity cohorts must also exclude every death.
            sens = pd.read_csv(result / 'cohorts' / f'{window}_any.csv')
            assert audit.loc[sens.hhidpn, 'alive_at_final_wave'].all()
            assert not audit.loc[sens.hhidpn, 'death_by_final_wave'].any()
            pred = [c for c in d if c not in ('hhidpn', 'demcls', 'y')]
            groups = {pre: [c for c in pred if c.startswith(pre)]
                      for pre in ('edstg', 'cendiv', 'mstat', 'raceeth', 'lbrf')}
            groups.update({c[:-6]: [c[:-6], c] for c in pred if c.endswith('dontdo')})
            grouped = set(c for v in groups.values() for c in v)
            groups.update({c: [c] for c in pred if c not in grouped})
            assert len(pred) == 94 and len(groups) == 60
            for v, cols in groups.items():
                missing.append(dict(baseline=year, horizon=h, variable=v,
                                    fraction_missing=d[cols].isna().any(axis=1).mean()))
            if h == 6:
                cols = [f'dementia_{year+k}' for k in (2, 4, 6)]
                paths = ['N' + ''.join('?' if pd.isna(v) else 'D' if v == 1 else 'N' for v in values)
                         for values in a[cols].to_numpy()]
                counts = pd.Series(paths).value_counts()
                sequences.extend(dict(baseline=year, horizon=h, sequence=s, n=int(n)) for s, n in counts.items())
            # Saved predictions must match the exact revised cohort, not just its size.
            for model in MODELS:
                if year != 2010:
                    continue
                p = pd.read_csv(result / f'predictions/final/h{h}/{model}/temporal_predictions.csv')
                assert p.hhidpn.is_unique and set(p.hhidpn) == set(d.hhidpn)
                expected = d.set_index('hhidpn').loc[p.hhidpn, 'y']
                assert np.array_equal(p.y, expected)
                assert np.isfinite(p.p).all() and p.p.between(0, 1).all()
            checks.append(dict(window=window, n=len(d), cases=int(d.y.sum()), deaths_included=0,
                               predictors=60, model_inputs=94, sensitivity_deaths_included=0))
    pd.DataFrame(sequences).to_csv(prep / 'fig1_observation/observation_sequences.csv', index=False)
    pd.DataFrame(missing).to_csv(prep / 'fig2_coverage/missingness_by_predictor.csv', index=False)
    for f in ['importance_permutation_h6.csv', 'importance_shap_h6.csv']:
        frame = pd.read_csv(tables / f)
        assert set(frame.model) == set(MODELS)
        assert frame.groupby('model').size().eq(60).all()
        shutil.copy2(tables / f, prep / 'fig4_signal' / f)
    shutil.copy2(tables / 'importance_meta_h6.csv', prep / 'fig4_signal/prediction_reconstruction_check.csv')
    for f in ['calibration_curves.csv', 'decision_curves.csv']:
        shutil.copy2(tables / f, prep / 'fig5_reliability' / f)
    for h in (6, 8):
        for m in ['LR', 'CatBoost', 'TabICL']:
            for f in ['selection.json', 'validation_grid.csv']:
                if not (result / f'figure_inputs/selection/final/h{h}/{m}' / f).is_file():
                    raise FileNotFoundError(f'Missing result/figure_inputs for Fig5: h{h}/{m}/{f}; rebuild result.zip with tools/package_result_model.py result')
    (out / 'qa/cohort_checks.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    return checks
