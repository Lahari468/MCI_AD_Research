"""
Participant-Level Paired Bootstrap Evaluation
Module: src/evaluation/bootstrap.py
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


def run_paired_bootstrap(
    y_true: np.ndarray,
    prob_test: np.ndarray,
    prob_ref: np.ndarray,
    num_bootstrap: int = 2000,
    random_seed: int = 42
) -> dict:
    """
    Evaluate participant-level paired bootstrap differences between two models.
    Calculates 95% empirical percentile confidence intervals and two-sided empirical tail p-values.
    """
    rng = np.random.default_rng(random_seed)
    n = len(y_true)
    delta_auc, delta_prauc, delta_brier = [], [], []

    for _ in range(num_bootstrap):
        idx = rng.integers(0, n, n)
        # Ensure resample contains both positive and negative cases
        if len(np.unique(y_true[idx])) > 1:
            d_a = roc_auc_score(y_true[idx], prob_test[idx]) - roc_auc_score(y_true[idx], prob_ref[idx])
            d_p = average_precision_score(y_true[idx], prob_test[idx]) - average_precision_score(y_true[idx], prob_ref[idx])
            d_b = brier_score_loss(y_true[idx], prob_test[idx]) - brier_score_loss(y_true[idx], prob_ref[idx])
            delta_auc.append(d_a)
            delta_prauc.append(d_p)
            delta_brier.append(d_b)

    ci_a = np.percentile(delta_auc, [2.5, 97.5])
    ci_p = np.percentile(delta_prauc, [2.5, 97.5])
    ci_b = np.percentile(delta_brier, [2.5, 97.5])

    # Empirical bootstrap-tail p-values: 2 * min(P(delta <= 0), P(delta >= 0))
    p_a = 2 * min(np.mean(np.array(delta_auc) <= 0), np.mean(np.array(delta_auc) >= 0))
    p_p = 2 * min(np.mean(np.array(delta_prauc) <= 0), np.mean(np.array(delta_prauc) >= 0))
    p_b = 2 * min(np.mean(np.array(delta_brier) <= 0), np.mean(np.array(delta_brier) >= 0))

    pt_d_a = roc_auc_score(y_true, prob_test) - roc_auc_score(y_true, prob_ref)
    pt_d_p = average_precision_score(y_true, prob_test) - average_precision_score(y_true, prob_ref)
    pt_d_b = brier_score_loss(y_true, prob_test) - brier_score_loss(y_true, prob_ref)

    return {
        'delta_auroc': round(float(pt_d_a), 4),
        'ci_auroc_lo': round(float(ci_a[0]), 4),
        'ci_auroc_hi': round(float(ci_a[1]), 4),
        'p_auroc': round(float(p_a), 4),
        'delta_prauc': round(float(pt_d_p), 4),
        'ci_prauc_lo': round(float(ci_p[0]), 4),
        'ci_prauc_hi': round(float(ci_p[1]), 4),
        'p_prauc': round(float(p_p), 4),
        'delta_brier': round(float(pt_d_b), 4),
        'ci_brier_lo': round(float(ci_b[0]), 4),
        'ci_brier_hi': round(float(ci_b[1]), 4),
        'p_brier': round(float(p_b), 4)
    }
