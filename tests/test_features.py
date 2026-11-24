"""Unit tests for clinical and FreeSurfer feature extraction and modality status."""

import numpy as np
import pandas as pd
import pytest

from src.features.clinical import clean_clinical_metric, CLINICAL_FULL, CLINICAL_NOCIRC
from src.features.freesurfer import FS_TARGET_MAPPING


def test_clinical_negative_recoding():
    """Verify that negative sentinel values recode to NaN with was_negative flag True."""
    val, was_neg = clean_clinical_metric(-1.0, "MMSCORE")
    assert np.isnan(val)
    assert was_neg is True

    val, was_neg = clean_clinical_metric(-4.0, "CDRSB")
    assert np.isnan(val)
    assert was_neg is True

    val, was_neg = clean_clinical_metric(28.0, "MMSCORE")
    assert val == 28.0
    assert was_neg is False

    val, was_neg = clean_clinical_metric(np.nan, "FAQTOTAL")
    assert np.isnan(val)
    assert was_neg is False


def test_clinical_feature_sets():
    """Verify definitions of clinical_full and clinical_nocirc."""
    assert len(CLINICAL_FULL) == 10
    assert len(CLINICAL_NOCIRC) == 7
    circular_feats = {"CDRSB", "CDGLOBAL", "FAQTOTAL"}
    assert circular_feats.issubset(set(CLINICAL_FULL))
    assert circular_feats.isdisjoint(set(CLINICAL_NOCIRC))


def test_freesurfer_verbatim_mapping():
    """Verify that all FreeSurfer thickness targets map to Thickness Average (TA)."""
    thickness_keys = ["ST24TA", "ST83TA", "ST44TA", "ST103TA", "ST40TA", "ST99TA", "ST52TA", "ST111TA"]
    for key in thickness_keys:
        assert key in FS_TARGET_MAPPING
        assert key.endswith("TA")
        assert "Thickness Average" in FS_TARGET_MAPPING[key]["datadic_text"]
        assert "Volume" not in FS_TARGET_MAPPING[key]["datadic_text"]

    # Verify hippocampal volumes map to Subcortical Volume (SV)
    assert FS_TARGET_MAPPING["ST29SV"]["datadic_text"].startswith("Subcortical Volume")
    assert FS_TARGET_MAPPING["ST88SV"]["datadic_text"].startswith("Subcortical Volume")


def test_modality_status_categories():
    """Verify categorical constraints in data/processed/modality_status.csv."""
    df = pd.read_csv("data/processed/modality_status.csv")
    assert set(df["clinical_status"].unique()).issubset({"OBSERVED", "PARTIAL"})
    assert set(df["freesurfer_status"].unique()).issubset({"OBSERVED", "PARTIAL", "ABSENT", "FAILED"})
    assert set(df["mri_status"].unique()).issubset({"OBSERVED_LOCAL", "AVAILABLE_NOT_DOWNLOADED", "ABSENT_NO_SCAN"})
