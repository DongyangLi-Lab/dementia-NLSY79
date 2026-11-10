"""Original KernelSHAP solve with each subject's recorded execution warmups.

This module performs no acceptance or import: callers must validate sources and
the applicable local/remote controls before publishing any results.
"""
import numpy as np


def solve_subjects(coalitions, subject_indices, records):
    """Return SHAP in the requested subject order, retaining input float dtype."""
    indices = list(subject_indices)
    if len(indices) != len(records) or len(indices) != len(set(indices)) or not indices:
        raise ValueError('Provide a nonempty list of unique subjects and matching records')
    z = np.asarray(coalitions)
    if z.shape != (800, 60) or not np.isin(z, [0, 1]).all():
        raise ValueError('Expected original 800-coalition, 60-group design')

    def logit(p):
        return np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

    z = z.astype(float)
    columns, deltas = [], []
    for index, record in zip(indices, records):
        if not isinstance(index, (int, np.integer)) or not 0 <= index < 500:
            raise ValueError('Subject outside the original 500-person design')
        p0, p1, predictions = (np.asarray(record[key]) for key in ('p0', 'p1', 'predictions'))
        for arr, shape in ((p0, (10,)), (p1, (500,)), (predictions, (1, 800, 10))):
            if (arr.shape != shape or not np.issubdtype(arr.dtype, np.floating)
                    or not np.isfinite(arr).all() or np.any((arr < 0) | (arr > 1))):
                raise ValueError(f'Invalid probabilities for subject {index}')
        v0 = logit(p0.mean())
        # Preserve the original scalar/array arithmetic. Under NumPy 1.x,
        # stacking scalar v0s first would promote float32 predictions to float64
        # before subtraction and change the original rounding path.
        delta = (logit(p1[index:index + 1]) - v0)[0]
        values = logit(predictions[0].mean(axis=1)) - v0
        columns.append(values - z[:, -1] * delta)
        deltas.append(delta)

    d = z[:, :-1] - z[:, [-1]]
    y = np.stack(columns, axis=1)
    phi = np.linalg.lstsq(d, y, rcond=None)[0].T
    phi = np.column_stack([phi, np.asarray(deltas) - phi.sum(axis=1)])
    if not np.isfinite(phi).all():
        raise ValueError('Nonfinite SHAP result')
    return phi
