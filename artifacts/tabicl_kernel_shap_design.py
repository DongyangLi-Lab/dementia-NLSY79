"""Save the exact original KernelSHAP random design for later resumable execution.

This writes indices and masks, not SHAP values or model predictions.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from artifact_store import job_dir, save_receipt, write_file
from inference import importance
from train.cohorts import ARTIFACTS, load_cohorts, sha256
from train.imputation import cache_stamp, read_cache, cache_path

def design(n_ext, n_dev, groups, columns, n_explain, n_background, n_coalitions, seed=2026):
    rng = np.random.default_rng(seed)
    names = list(groups)
    m = len(names)
    explain = rng.choice(n_ext, n_explain, replace=False)
    background = rng.choice(n_dev, n_background, replace=False)
    sizes = np.arange(1, m)
    weights = (m - 1) / (sizes * (m - sizes))
    z = np.zeros((n_coalitions, m), bool)
    for i in range(0, n_coalitions, 2):
        z[i, rng.choice(m, rng.choice(sizes, p=weights / weights.sum()), replace=False)] = True
        z[i + 1] = ~z[i]
    colmask = np.zeros((n_coalitions, len(columns)), bool)
    for g, name in enumerate(names):
        colmask[:, [columns.index(c) for c in groups[name]]] = z[:, [g]]
    return explain, background, z, colmask

if __name__ == '__main__':
    _, ext, predictors = load_cohorts(6)
    cache = read_cache(6, 'final', 'external', cache_stamp(6, 'final'))
    x_dev, x_ext = cache['X_dev'], cache['X_ext']
    if len(ext) != len(x_ext) or list(x_dev.columns) != list(x_ext.columns):
        raise ValueError('Cohort and imputation cache are not aligned')
    cols = list(x_ext.columns)
    groups = importance.predictor_groups(predictors)
    assert len(groups) == 60
    explain, background, z, mask = design(len(x_ext), len(x_dev), groups, cols,
        importance.KS_EXPLAIN, importance.KS_BACKGROUND, importance.KS_COALITIONS)
    if not np.array_equal(z[1::2], ~z[::2]):
        raise ValueError('Coalition pairs are not complementary')
    output = ARTIFACTS / 'tabicl_kernel_shap_design.npz'
    def save(path):
        with path.open('wb') as stream:
            np.savez_compressed(stream, explain_indices=explain, background_indices=background,
                                coalitions=z, column_mask=mask, groups=np.array(list(groups)),
                                columns=np.array(cols))
    write_file(output, save)
    stamp = {'cache': sha256(cache_path(6, 'final', 'external')),
             'importance': sha256(Path(importance.__file__)), 'design': sha256(Path(__file__)),
             'inference': sha256(job_dir('predictions', 6, 'final', 'TabICL') / 'inference.done.json')}
    save_receipt(output.with_suffix('.done.json'), stamp, (output,))
    print(f'Prepared {len(explain)} participants, {len(background)} background rows and {len(z)} coalitions', flush=True)
