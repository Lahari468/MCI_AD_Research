"""
Tests for Stage 5 exploratory experiment outputs (E1 and E2).
Verification of output schemas, dimensions, test-set leak-free guards, and registry hashes.
"""

import os
import hashlib
import numpy as np
import pandas as pd
import pytest

@pytest.fixture(scope="module")
def locked_test_rids():
    sp = pd.read_csv("data/processed/splits/splits_v1.csv")
    test_rids = set(sp[sp.split == "test"].rid.unique())
    assert len(test_rids) == 156, f"Expected 156 locked test RIDs, got {len(test_rids)}"
    return test_rids

def test_e1_oof_predictions(locked_test_rids):
    path = "results/experiments/e1_oof.csv"
    assert os.path.exists(path), f"Missing {path}"
    df = pd.read_csv(path)
    
    # 11 models x 10 repeats x 564 participants = 62,040 rows
    assert len(df) == 62040, f"Expected 62040 rows in e1_oof, got {len(df)}"
    assert df["model"].nunique() == 11, f"Expected 11 models, got {df['model'].nunique()}"
    assert df["repeat"].nunique() == 10, f"Expected 10 repeats, got {df['repeat'].nunique()}"
    assert df["rid"].nunique() == 564, f"Expected 564 participants, got {df['rid'].nunique()}"
    
    # Check no NaN values
    assert not df.isna().any().any(), "e1_oof contains NaN values"
    
    # Check no locked-test RIDs
    oof_rids = set(df["rid"].unique())
    leak = oof_rids & locked_test_rids
    assert len(leak) == 0, f"CRITICAL LEAK: locked test RIDs found in e1_oof: {leak}"

def test_e1_folds_leak_free(locked_test_rids):
    path = "results/experiments/e1_folds.csv"
    assert os.path.exists(path), f"Missing {path}"
    df = pd.read_csv(path)
    
    fold_rids = set(df["rid"].unique())
    leak = fold_rids & locked_test_rids
    assert len(leak) == 0, f"CRITICAL LEAK: locked test RIDs found in e1_folds: {leak}"

def test_permutation_null_arrays():
    p1 = "results/experiments/e1_permutation_null_c_roi.npy"
    p2 = "results/experiments/e1_permutation_null_prot_roi.npy"
    
    assert os.path.exists(p1), f"Missing {p1}"
    assert os.path.exists(p2), f"Missing {p2}"
    
    arr1 = np.load(p1)
    arr2 = np.load(p2)
    
    assert arr1.shape == (500,), f"Expected shape (500,), got {arr1.shape}"
    assert arr2.shape == (500,), f"Expected shape (500,), got {arr2.shape}"
    assert not np.isnan(arr1).any(), "NaN found in permutation null 1"
    assert not np.isnan(arr2).any(), "NaN found in permutation null 2"

def test_e1_summary_structure():
    path = "results/experiments/e1_summary.csv"
    assert os.path.exists(path), f"Missing {path}"
    df = pd.read_csv(path)
    
    model_rows = df[df["section"] == "model_performance"]
    contrast_rows = df[df["section"] == "estimand_contrast"]
    perm_rows = df[df["section"] == "permutation_test"]
    
    assert len(model_rows) == 11, f"Expected 11 model rows, got {len(model_rows)}"
    assert len(contrast_rows) == 11, f"Expected 11 contrast rows, got {len(contrast_rows)}"
    assert len(perm_rows) == 2, f"Expected 2 permutation rows, got {len(perm_rows)}"
    assert len(df) == 24, f"Expected 24 total rows, got {len(df)}"

def test_registry_hashes():
    reg = pd.read_csv("reports/audit/plan_registry.csv")
    
    files_to_check = [
        "reports/audit/09c_experiment_plan.md",
        "scripts/experiments/e1_specificity.py",
        "scripts/experiments/e2_switch.py"
    ]
    
    for fpath in files_to_check:
        assert os.path.exists(fpath), f"Missing {fpath}"
        current_hash = hashlib.sha256(open(fpath, "rb").read()).hexdigest()
        
        sub = reg[reg["file"] == fpath]
        assert len(sub) > 0, f"No registry entry for {fpath}"
        latest_reg_hash = sub.iloc[-1]["sha256"]
        
        assert current_hash == latest_reg_hash, (
            f"Hash mismatch for {fpath}: current={current_hash}, registry={latest_reg_hash}"
        )

def test_e2_outputs_exist():
    e2_files = [
        "results/experiments/e2_switch_curve.csv",
        "results/experiments/e2_switch_paired_cis.csv",
        "results/experiments/F_switch_curve.png",
        "results/experiments/e2_stdout.txt"
    ]
    for p in e2_files:
        assert os.path.exists(p), f"Missing E2 output file: {p}"
        assert os.path.getsize(p) > 0, f"E2 output file is empty: {p}"
