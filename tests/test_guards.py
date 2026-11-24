"""Data integrity and locked test evaluation guard tests."""

import os
import subprocess
import sys
import pandas as pd
import pytest


def test_train_final_refuses_without_final_eval():
    """Verify that train_final.py with --final refuses to run unless FINAL_EVAL=1."""
    env = {k: v for k, v in os.environ.items() if k != "FINAL_EVAL"}
    cmd = [
        sys.executable,
        "src/training/train_final.py",
        "--name",
        "smoke",
        "--final",
    ]
    proc = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0, "train_final.py must fail when FINAL_EVAL is not set to 1"
    assert (
        "FINAL_EVAL=1" in proc.stderr or "AssertionError" in proc.stderr
    ), f"Expected assertion error about FINAL_EVAL=1, got: {proc.stderr}"


def test_no_rid_overlap_between_test_and_dev_folds():
    """Verify zero RID overlap between locked test and development folds."""
    splits_df = pd.read_csv("data/processed/splits/splits_v1.csv")

    test_rids = set(splits_df[splits_df["split"] == "test"]["rid"].unique())
    dev_rids = set(splits_df[splits_df["split"] == "dev"]["rid"].unique())

    # Overall test vs dev overlap
    assert len(test_rids) > 0, "Test set cannot be empty"
    assert len(dev_rids) > 0, "Dev set cannot be empty"
    overlap = test_rids.intersection(dev_rids)
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping RIDs between test and dev sets."

    # Per-fold test vs dev fold overlap
    dev_df = splits_df[splits_df["split"] == "dev"]
    for (repeat, fold), group in dev_df.groupby(["repeat", "fold"]):
        fold_rids = set(group["rid"].unique())
        fold_overlap = test_rids.intersection(fold_rids)
        assert len(fold_overlap) == 0, (
            f"Found {len(fold_overlap)} overlapping RIDs between test and repeat {repeat} fold {fold}."
        )


def test_cohort_v2b_eligible_labels_are_zero_or_one():
    """Verify that every label_primary in cohort_v2b.csv among eligible participants is 0 or 1."""
    cohort_df = pd.read_csv("data/processed/cohort_v2b.csv")

    eligible_df = cohort_df[cohort_df["exclusion_reason"] == "ELIGIBLE"]
    assert len(eligible_df) > 0, "No eligible participants found in cohort_v2b.csv"
    assert eligible_df["label_primary"].isin([0, 1]).all(), (
        "All eligible participants must have label_primary in {0, 1}"
    )
    assert not eligible_df["label_primary"].isna().any(), (
        "Eligible participants must not have missing label_primary"
    )

    labeled_df = cohort_df[cohort_df["label_primary"].notna()]
    assert labeled_df["label_primary"].isin([0, 1]).all(), (
        "All labeled participants must have label_primary in {0, 1}"
    )
    assert len(eligible_df) == len(labeled_df), (
        f"Mismatch between eligible ({len(eligible_df)}) and non-null label ({len(labeled_df)}) participants"
    )
