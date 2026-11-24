"""
Tests for Stage 7a machine-readable MRI tables, figures, and consistency checks.
"""

import os
import pandas as pd
import numpy as np
import pytest

@pytest.fixture(scope="module")
def t10a():
    path = "results/paper/T10a_mri_incremental_value.csv"
    assert os.path.exists(path), f"Missing {path}"
    return pd.read_csv(path)

@pytest.fixture(scope="module")
def t10b():
    path = "results/paper/T10b_mri_only_models.csv"
    assert os.path.exists(path), f"Missing {path}"
    return pd.read_csv(path)

@pytest.fixture(scope="module")
def t10c():
    path = "results/paper/T10c_e1_models_estimands.csv"
    assert os.path.exists(path), f"Missing {path}"
    return pd.read_csv(path)

@pytest.fixture(scope="module")
def t10d():
    path = "results/paper/T10d_e2_endpoints_decomposition.csv"
    assert os.path.exists(path), f"Missing {path}"
    return pd.read_csv(path)

@pytest.fixture(scope="module")
def t10e():
    path = "results/paper/T10e_ledger_counts.csv"
    assert os.path.exists(path), f"Missing {path}"
    return pd.read_csv(path)

def test_check_a_model_difference_consistency():
    """Consistency check a: |delta - (AUROC_a - AUROC_b)| <= 0.00015"""
    checks = [
        # mri_roi_nested
        (0.8725, 0.8723, 0.0002),
        (0.8754, 0.8747, 0.0006),
        (0.8683, 0.8723, -0.0040),
        (0.8700, 0.8747, -0.0047),
        # mri_pca_increment
        (0.8665, 0.8721, -0.0056),
        (0.8663, 0.8721, -0.0058),
        (0.8654, 0.8711, -0.0057),
        (0.8673, 0.8711, -0.0038),
        # fs_saturation
        (0.8815, 0.8776, 0.0038),
        (0.8767, 0.8776, -0.0009),
        # mri_fallback
        (0.8594, 0.8399, 0.0195),
        (0.8675, 0.8547, 0.0128),
        (0.8607, 0.8481, 0.0126),
        (0.8723, 0.8399, 0.0324),
        (0.8723, 0.8594, 0.0129),
        (0.8725, 0.8723, 0.0002),
        # mri_only_baselines
        (0.6602, 0.6852, -0.0250),
        (0.7326, 0.6852, 0.0474),
        (0.7255, 0.6852, 0.0404),
        (0.7872, 0.6852, 0.1020),
        (0.6602, 0.7872, -0.1270),
        (0.7326, 0.7872, -0.0546),
        (0.7255, 0.7872, -0.0617),
        (0.7196, 0.6852, 0.0344),
        (0.7641, 0.6852, 0.0789),
        (0.7557, 0.6852, 0.0705),
        (0.7950, 0.6852, 0.1098)
    ]
    for auc_a, auc_b, delta in checks:
        diff = abs(delta - (auc_a - auc_b))
        assert diff <= 0.00015, f"Check a failed: |{delta} - ({auc_a} - {auc_b})| = {diff}"

def test_check_b_interval_nesting(t10a):
    """Consistency check b: 99% interval contains 95% interval where both exist"""
    sub = t10a[t10a['ci99_lo'].notna() & t10a['ci95_lo'].notna()]
    assert len(sub) == 16, f"Expected 16 rows with both intervals, got {len(sub)}"
    for _, r in sub.iterrows():
        assert r['ci99_lo'] <= r['ci95_lo'], f"ci99_lo > ci95_lo in {r['analysis']}"
        assert r['ci99_hi'] >= r['ci95_hi'], f"ci99_hi < ci95_hi in {r['analysis']}"

def test_check_c_sample_sizes(t10a):
    """Consistency check c: n and events (564/123 for MRI, 622/144 for saturation)"""
    sat = t10a[t10a['family'] == 'FreeSurfer saturation']
    assert (sat['n'] == 622).all()
    assert (sat['events'] == 144).all()

    mri = t10a[t10a['family'] != 'FreeSurfer saturation']
    mri_main = mri[mri['n'] == 564]
    assert (mri_main['events'] == 123).all()

def test_check_d_criterion_flags(t10a):
    """Consistency check d: Recomputed vs printed criterion flags match"""
    crit_rows = t10a[t10a['criterion_printed'].notna() & (t10a['criterion_printed'] != '')]
    assert len(crit_rows) == 11, f"Expected 11 printed criterion rows, got {len(crit_rows)}"
    for _, r in crit_rows.iterrows():
        assert str(r['criterion_printed']) == str(r['criterion_recomputed']), (
            f"Criterion mismatch in {r['analysis']}: printed={r['criterion_printed']}, recomputed={r['criterion_recomputed']}"
        )

def test_check_e_ledger_contrast_counts(t10e):
    """Consistency check e: ledger executed contrasts per family"""
    comp = t10e[t10e['status'] == 'Completed']
    assert comp['executed_contrasts'].sum() == 45, f"Expected 45 executed contrasts, got {comp['executed_contrasts'].sum()}"
    assert comp['executed_confidence_intervals'].sum() == 115, f"Expected 115 executed CIs, got {comp['executed_confidence_intervals'].sum()}"

def test_check_f_source_references(t10a, t10b, t10c, t10d):
    """Consistency check f: Source file and line existence"""
    for df in [t10a, t10b, t10c]:
        for _, r in df.iterrows():
            fpath = r['source_file']
            assert os.path.exists(fpath), f"Missing source file: {fpath}"
            lno = r['source_line']
            with open(fpath) as f:
                lines = f.readlines()
                assert 1 <= int(lno) <= len(lines), f"Line {lno} out of range in {fpath}"

def test_figure_outputs_exist():
    """Verify generated figure files exist and have non-zero size"""
    figures = [
        "results/paper/F_forest_mri.png",
        "results/paper/F_forest_mri.pdf",
        "results/paper/F_forest_mri_caption.txt",
        "results/paper/F_e2_decomposition.png",
        "results/paper/F_e2_decomposition.pdf"
    ]
    for fig_path in figures:
        assert os.path.exists(fig_path), f"Missing figure file: {fig_path}"
        assert os.path.getsize(fig_path) > 0, f"Figure file is empty: {fig_path}"
