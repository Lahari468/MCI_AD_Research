"""
Evaluation Metrics Computation
Module: src/evaluation/metrics.py
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Compute comprehensive discrimination, calibration, and classification metrics.
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    auroc = roc_auc_score(y_true, y_prob)
    prauc = average_precision_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    brier = brier_score_loss(y_true, y_prob)

    return {
        'auroc': float(auroc),
        'prauc': float(prauc),
        'accuracy': float(acc),
        'balanced_accuracy': float(bacc),
        'sensitivity': float(sens),
        'specificity': float(spec),
        'precision': float(prec),
        'f1': float(f1),
        'brier': float(brier),
        'threshold': float(threshold),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn)
    }
