"""
Unit tests for evaluation metrics and paired bootstrap calculation.
"""

import numpy as np
from src.evaluation.metrics import compute_metrics
from src.evaluation.bootstrap import run_paired_bootstrap


def test_compute_metrics():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])

    m = compute_metrics(y_true, y_prob, threshold=0.5)
    assert m['auroc'] == 1.0, f"Expected perfect AUROC 1.0, got {m['auroc']}"
    assert m['accuracy'] == 1.0
    assert m['sensitivity'] == 1.0
    assert m['specificity'] == 1.0
    assert m['tp'] == 3
    assert m['tn'] == 3


def test_paired_bootstrap():
    rng = np.random.default_rng(42)
    y_true = rng.choice([0, 1], size=100, p=[0.75, 0.25])
    p_ref = rng.uniform(0.1, 0.6, size=100)
    p_test = p_ref + rng.normal(0.05, 0.02, size=100)
    p_test = np.clip(p_test, 0.0, 1.0)

    res = run_paired_bootstrap(y_true, p_test, p_ref, num_bootstrap=100, random_seed=42)
    assert 'delta_auroc' in res
    assert 'ci_auroc_lo' in res
    assert 'ci_auroc_hi' in res
    assert 'p_auroc' in res
    assert res['ci_auroc_lo'] <= res['ci_auroc_hi']
