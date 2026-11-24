"""
Unit tests verifying cohort definition integrity, split disjointness, and isolation.
"""

import os
import pandas as pd
import pytest


def test_development_holdout_isolation():
    splits_path = "data/processed/splits/splits_v1.csv"
    if not os.path.exists(splits_path):
        pytest.skip("Splits file not present; skipping local data test.")

    sp = pd.read_csv(splits_path)
    test_rids = set(sp[sp.split == "test"]["rid"].unique())
    dev_rids = set(sp[sp.split != "test"]["rid"].unique())

    assert len(test_rids) == 156, f"Expected 156 locked holdout cases, got {len(test_rids)}"
    assert len(dev_rids) == 622, f"Expected 622 development cases, got {len(dev_rids)}"
    assert len(test_rids & dev_rids) == 0, "CRITICAL ERROR: Overlap between dev and test sets!"


def test_cross_validation_split_structure():
    splits_path = "data/processed/splits/splits_v1.csv"
    if not os.path.exists(splits_path):
        pytest.skip("Splits file not present; skipping local data test.")

    sp = pd.read_csv(splits_path)
    dev_sp = sp[sp.split != "test"]

    assert set(dev_sp["repeat"].unique()) == {0, 1, 2}, "Expected exactly 3 repeats (0, 1, 2)"
    for rep in [0, 1, 2]:
        rep_folds = dev_sp[dev_sp.repeat == rep]["fold"].unique()
        assert len(rep_folds) == 5, f"Expected 5 folds in repeat {rep}, got {len(rep_folds)}"
