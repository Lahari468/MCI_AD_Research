"""Unit tests for dataset splitting and secure split loader."""

import os
import pandas as pd
import pytest

from src.data.splits import load_dev_fold, load_locked_test


def test_zero_rid_overlap():
    """Verify zero participant overlap between locked test and dev sets, and across CV folds."""
    splits_df = pd.read_csv("data/processed/splits/splits_v1.csv")

    test_rids = set(splits_df[splits_df["split"] == "test"]["rid"].unique())
    dev_rids = set(splits_df[splits_df["split"] == "dev"]["rid"].unique())

    # 1. Zero overlap between dev and test
    overlap = test_rids.intersection(dev_rids)
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping RIDs between dev and test."
    assert len(test_rids) == 156
    assert len(dev_rids) == 622

    # 2. Zero overlap across validation folds within each CV repeat
    for rep in range(3):
        dev_rep = splits_df[(splits_df["split"] == "dev") & (splits_df["repeat"] == rep)]
        fold_sets = [set(dev_rep[dev_rep["fold"] == f]["rid"]) for f in range(5)]
        for f1 in range(5):
            for f2 in range(f1 + 1, 5):
                inter = fold_sets[f1].intersection(fold_sets[f2])
                assert len(inter) == 0, f"Overlap between fold {f1} and {f2} in repeat {rep}."

        # Union of all 5 folds must equal the full dev set
        union_all = set().union(*fold_sets)
        assert union_all == dev_rids


def test_fold_prevalence_within_three_points():
    """Verify that prevalence across all CV folds is within 3 percentage points of dev prevalence."""
    cohort = pd.read_csv("data/processed/cohort_v2b.csv")
    label_map = cohort.set_index("RID")["label_primary"].to_dict()

    splits_df = pd.read_csv("data/processed/splits/splits_v1.csv")
    dev_df = splits_df[splits_df["split"] == "dev"].copy()
    dev_df["label"] = dev_df["rid"].map(label_map)

    overall_dev_prev = dev_df.drop_duplicates("rid")["label"].mean()

    for rep in range(3):
        for fold in range(5):
            fold_labels = dev_df[(dev_df["repeat"] == rep) & (dev_df["fold"] == fold)]["label"]
            fold_prev = fold_labels.mean()
            diff_points = abs(fold_prev - overall_dev_prev) * 100.0
            assert diff_points < 3.0, f"Repeat {rep} Fold {fold} prevalence {fold_prev:.4f} exceeds 3 points from dev {overall_dev_prev:.4f}"


def test_locked_test_guard():
    """Verify locked test set raises PermissionError unless FINAL_EVAL=1."""
    if "FINAL_EVAL" in os.environ:
        del os.environ["FINAL_EVAL"]

    with pytest.raises(PermissionError):
        load_locked_test()

    os.environ["FINAL_EVAL"] = "0"
    with pytest.raises(PermissionError):
        load_locked_test()

    os.environ["FINAL_EVAL"] = "1"
    test_rids = load_locked_test()
    assert len(test_rids) == 156
    del os.environ["FINAL_EVAL"]


def test_load_dev_fold():
    """Verify load_dev_fold returns partitioned train and val RIDs."""
    train_rids, val_rids = load_dev_fold(repeat=0, fold=0)
    assert len(train_rids) + len(val_rids) == 622
    assert len(set(train_rids).intersection(set(val_rids))) == 0
    assert len(val_rids) == 125
