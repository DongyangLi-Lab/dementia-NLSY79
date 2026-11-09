"""Continuous-score evaluation and paired household bootstrap."""
import numpy as np


def metrics(y, prediction, weights=None):
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    error = prediction-y
    mse = np.average(error**2, weights=weights)
    variance = np.average((y-np.average(y, weights=weights))**2, weights=weights)
    design = np.column_stack([np.ones(len(y)), prediction])
    w = np.ones(len(y)) if weights is None else np.asarray(weights)
    beta = np.linalg.lstsq(design*np.sqrt(w[:, None]), y*np.sqrt(w), rcond=None)[0]
    return {'n': len(y), 'RMSE': float(np.sqrt(mse)), 'MAE': float(np.average(abs(error), weights=weights)),
            'R2': float(1-mse/variance) if variance > 0 else None,
            'mean_prediction_error': float(np.average(error, weights=weights)),
            'calibration_intercept': float(beta[0]) if np.ptp(prediction)>0 else None,
            'calibration_slope': float(beta[1]) if np.ptp(prediction)>0 else None}


def paired_bootstrap(y, a, b, households, repeats=2000, seed=42):
    """Resample whole target households, preserving paired A/B predictions."""
    y, a, b, households = map(np.asarray, (y, a, b, households))
    groups = np.unique(households)
    indices = [np.flatnonzero(households == group) for group in groups]
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(repeats):
        ids = np.concatenate([indices[i] for i in rng.integers(0, len(groups), len(groups))])
        differences.append(float(np.sqrt(np.mean((y[ids]-a[ids])**2)) - np.sqrt(np.mean((y[ids]-b[ids])**2))))
    return {'RMSE_A_minus_B': float(np.sqrt(np.mean((y-a)**2))-np.sqrt(np.mean((y-b)**2))),
            'CI95_low': float(np.quantile(differences, .025)), 'CI95_high': float(np.quantile(differences, .975)),
            'bootstrap_households': len(groups), 'bootstrap_repeats': repeats, 'seed': seed}
