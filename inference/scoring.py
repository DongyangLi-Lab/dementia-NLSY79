"""Prediction and metric helpers shared by tuning and held-out inference."""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)

def best_f1_threshold(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return float(thresholds[np.argmax(f1[:-1])])


def metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    return {'n': int(len(y)), 'cases': int(y.sum()), 'AUROC': roc_auc_score(y, p), 'AUPRC': average_precision_score(y, p),
            'Accuracy': accuracy_score(y, pred), 'Precision': precision_score(y, pred, zero_division=0),
            'Recall': recall_score(y, pred), 'F1': f1_score(y, pred), 'Brier': brier_score_loss(y, p),
            'threshold': threshold}



def predict(model, frame):
    p = np.asarray(model.predict_proba(frame.astype(np.float32)))[:, 1]
    if len(p) != len(frame) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Estimator returned invalid dementia probabilities")
    return p
