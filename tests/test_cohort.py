"""Unit tests for MCI to AD cohort construction with synthetic timelines.
Tests Cohort v1 and Cohort v2 (Amendments 1a-1j):
- Reversion (MCI -> CN)
- Censoring (< 640 d)
- Competing events (Codes 4 and 5)
- Missing diagnosis handling
- Conflicting sources (matched vs unmatched)
- AD on or before predictor cutoff
- Visits at exactly days 639, 640, 730, and 820
- Scans at exactly anchor - 180 d and anchor + 90 d, and window boundary assertions
- Missing core clinical baseline
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.cohort.build_cohort import evaluate_participant_timeline, evaluate_participant_timeline_v2


@pytest.fixture
def base_config() -> dict:
    return {
        "prediction_horizon_days": 730,
        "primary_negative_min_days": 640,
        "sens_pos_max_days": 820,
        "sens_neg_ge550_days": 550,
        "sens_neg_symm90_min_days": 640,
        "sens_neg_symm90_max_days": 820,
        "sens_neg_symm180_min_days": 550,
        "sens_neg_symm180_max_days": 910,
        "sens_neg_ge730_days": 730,
        "crosscheck_horizon_days": 910,
    }


# ============================================================================
# Cohort v1 Tests (Preserved)
# ============================================================================

def test_v1_primary_positive_conversion(base_config):
    """v1: Participant converts to AD within 730 days."""
    idx_date = pd.Timestamp("2020-01-01")
    phc_sub = pd.DataFrame([
        {"RID": 1, "EXAMDATE": idx_date, "VISCODE2": "bl", "PHC_Diagnosis": 2},
        {"RID": 1, "EXAMDATE": idx_date + pd.Timedelta(days=365), "VISCODE2": "m12", "PHC_Diagnosis": 3},
    ])
    dxsum_sub = pd.DataFrame([
        {"RID": 1, "EXAMDATE": idx_date, "dx_code": 2},
        {"RID": 1, "EXAMDATE": idx_date + pd.Timedelta(days=365), "dx_code": 3},
    ])

    res = evaluate_participant_timeline(
        rid=1,
        idx_date=idx_date,
        phc_sub=phc_sub,
        dxsum_sub=dxsum_sub,
        clin_dates=[idx_date],
        fs_dates=[idx_date],
        mri_dates=[idx_date],
        c_cfg=base_config,
    )

    assert res["label_primary"] == 1
    assert res["exclusion_reason"] == "ELIGIBLE"
    assert res["non_ad_flag"] is False
    assert res["conflict_flag"] is False


def test_v1_primary_negative_stable(base_config):
    """v1: Participant remains MCI at day 700 without AD conversion."""
    idx_date = pd.Timestamp("2020-01-01")
    phc_sub = pd.DataFrame([
        {"RID": 2, "EXAMDATE": idx_date, "VISCODE2": "bl", "PHC_Diagnosis": 2},
        {"RID": 2, "EXAMDATE": idx_date + pd.Timedelta(days=700), "VISCODE2": "m24", "PHC_Diagnosis": 2},
    ])
    dxsum_sub = pd.DataFrame([
        {"RID": 2, "EXAMDATE": idx_date, "dx_code": 2},
        {"RID": 2, "EXAMDATE": idx_date + pd.Timedelta(days=700), "dx_code": 2},
    ])

    res = evaluate_participant_timeline(
        rid=2,
        idx_date=idx_date,
        phc_sub=phc_sub,
        dxsum_sub=dxsum_sub,
        clin_dates=[idx_date],
        fs_dates=[idx_date],
        mri_dates=[idx_date],
        c_cfg=base_config,
    )

    assert res["label_primary"] == 0
    assert res["exclusion_reason"] == "ELIGIBLE"
    assert res["reverter_flag"] is False


# ============================================================================
# Cohort v2 Tests (Amendments 1a - 1j)
# ============================================================================

def test_v2_reversion_timeline(base_config):
    """v2: Reverter (MCI -> CN) post-anchor is flagged, not excluded."""
    idx_date = pd.Timestamp("2020-01-01")
    phc_sub = pd.DataFrame([
        {"RID": 201, "EXAMDATE": idx_date, "VISCODE2": "bl", "PHC_Diagnosis": 2},
        {"RID": 201, "EXAMDATE": idx_date + pd.Timedelta(days=365), "VISCODE2": "m12", "PHC_Diagnosis": 1},
        {"RID": 201, "EXAMDATE": idx_date + pd.Timedelta(days=700), "VISCODE2": "m24", "PHC_Diagnosis": 1},
    ])
    dxsum_sub = pd.DataFrame([
        {"RID": 201, "EXAMDATE": idx_date, "dx_code": 2},
        {"RID": 201, "EXAMDATE": idx_date + pd.Timedelta(days=700), "dx_code": 1},
    ])

    res = evaluate_participant_timeline_v2(
        rid=201,
        idx_date=idx_date,
        phc_sub=phc_sub,
        dxsum_sub=dxsum_sub,
        clin_dates=[idx_date],
        fs_dates=[idx_date],
        mri_dates=[idx_date],
        c_cfg=base_config,
    )

    assert res["reverter_flag"] is True
    assert res["label_primary"] == 0
    assert res["exclusion_reason"] == "ELIGIBLE"


def test_v2_censoring_insufficient_followup(base_config):
    """v2: Lost to follow-up at day 365 (< 640 d) without AD is excluded."""
    idx_date = pd.Timestamp("2020-01-01")
    phc_sub = pd.DataFrame([
        {"RID": 202, "EXAMDATE": idx_date, "VISCODE2": "bl", "PHC_Diagnosis": 2},
        {"RID": 202, "EXAMDATE": idx_date + pd.Timedelta(days=365), "VISCODE2": "m12", "PHC_Diagnosis": 2},
    ])
    dxsum_sub = pd.DataFrame([
        {"RID": 202, "EXAMDATE": idx_date, "dx_code": 2},
    ])

    res = evaluate_participant_timeline_v2(
        rid=202,
        idx_date=idx_date,
        phc_sub=phc_sub,
        dxsum_sub=dxsum_sub,
        clin_dates=[idx_date],
        fs_dates=[idx_date],
        mri_dates=[idx_date],
        c_cfg=base_config,
    )

    assert np.isnan(res["label_primary"])
    assert res["exclusion_reason"] == "INSUFFICIENT_FOLLOWUP"


def test_v2_competing_code4_non_ad(base_config):
    """v2: PHC_Diagnosis == 4 (non-AD dementia) within (T_anchor, T_anchor + 910 d] excludes."""
    idx_date = pd.Timestamp("2020-01-01")
    phc_sub = pd.DataFrame([
        {"RID": 203, "EXAMDATE": idx_date, "VISCODE2": "bl", "PHC_Diagnosis": 2},
        {"RID": 203, "EXAMDATE": idx_date + pd.Timedelta(days=365), "VISCODE2": "m12", "PHC_Diagnosis": 4},
    ])
    dxsum_sub = pd.DataFrame([
        {"RID": 203, "EXAMDATE": idx_date, "dx_code": 2},
    ])

    res = evaluate_participant_timeline_v2(
        rid=203,
        idx_date=idx_date,
        phc_sub=phc_sub,
        dxsum_sub=dxsum_sub,
        clin_dates=[idx_date],
        fs_dates=[idx_date],
        mri_dates=[idx_date],
        c_cfg=base_config,
    )

    assert res["non_ad_or_other_flag"] is True
    assert res["exclusion_reason"] == "NON_AD_OR_OTHER_DEMENTIA"
    assert np.isnan(res["label_primary"])


def test_v2_competing_code5_other_diagnosis(base_config):
    """v2: PHC_Diagnosis == 5 (Other Cognitive Diagnosis) within (T_anchor, T_anchor + 910 d] excludes."""
    idx_date = pd.Timestamp("2020-01-01")
    phc_sub = pd.DataFrame([
        {"RID": 204, "EXAMDATE": idx_date, "VISCODE2": "bl", "PHC_Diagnosis": 2},
        {"RID": 204, "EXAMDATE": idx_date + pd.Timedelta(days=500), "VISCODE2": "m18", "PHC_Diagnosis": 5},
    ])
    dxsum_sub = pd.DataFrame([
        {"RID": 204, "EXAMDATE": idx_date, "dx_code": 2},
    ])

    res = evaluate_participant_timeline_v2(
        rid=204,
        idx_date=idx_date,
        phc_sub=phc_sub,
        dxsum_sub=dxsum_sub,
        clin_dates=[idx_date],
        fs_dates=[idx_date],
        mri_dates=[idx_date],
        c_cfg=base_config,
    )

    assert res["non_ad_or_other_flag"] is True
    assert res["exclusion_reason"] == "NON_AD_OR_OTHER_DEMENTIA"
    assert np.isnan(res["label_primary"])


def test_v2_missing_diagnosis_cannot_confirm_negative(base_config):
    """v2: Missing PHC_Diagnosis (NaN) is never evidence of non-progression."""
    idx_date = pd.Timestamp("2020-01-01")
    # Follow-up visit at day 700 exists, but diagnosis is NaN
    phc_sub = pd.DataFrame([
        {"RID": 205, "EXAMDATE": idx_date, "VISCODE2": "bl", "PHC_Diagnosis": 2},
        {"RID": 205, "EXAMDATE": idx_date + pd.Timedelta(days=700), "VISCODE2": "m24", "PHC_Diagnosis": np.nan},
    ])
    dxsum_sub = pd.DataFrame([
        {"RID": 205, "EXAMDATE": idx_date, "dx_code": 2},
    ])

    res = evaluate_participant_timeline_v2(
        rid=205,
        idx_date=idx_date,
        phc_sub=phc_sub,
        dxsum_sub=dxsum_sub,
        clin_dates=[idx_date],
        fs_dates=[idx_date],
        mri_dates=[idx_date],
        c_cfg=base_config,
    )

    # Cannot be confirmed negative because diagnosis is NaN -> INSUFFICIENT_FOLLOWUP
    assert res["exclusion_reason"] == "INSUFFICIENT_FOLLOWUP"
    assert np.isnan(res["label_primary"])


def test_v2_conflicting_matched_sources(base_config):
    """v2: Disagreeing matched visit in [T_anchor, T_anchor + 910 d] excludes."""
    idx_date = pd.Timestamp("2020-01-01")
    fu_date = idx_date + pd.Timedelta(days=700)
    phc_sub = pd.DataFrame([
        {"RID": 206, "EXAMDATE": idx_date, "VISCODE2": "bl", "PHC_Diagnosis": 2},
        {"RID": 206, "EXAMDATE": fu_date, "VISCODE2": "m24", "PHC_Diagnosis": 2},
    ])
    # DXSUM says CN (1) on the same date while PHC says MCI (2)
    dxsum_sub = pd.DataFrame([
        {"RID": 206, "EXAMDATE": idx_date, "dx_code": 2},
        {"RID": 206, "EXAMDATE": fu_date, "dx_code": 1},
    ])

    res = evaluate_participant_timeline_v2(
        rid=206,
        idx_date=idx_date,
        phc_sub=phc_sub,
        dxsum_sub=dxsum_sub,
        clin_dates=[idx_date],
        fs_dates=[idx_date],
        mri_dates=[idx_date],
        c_cfg=base_config,
    )

    assert res["conflict_flag"] is True
    assert res["exclusion_reason"] == "PHC_DXSUM_CONFLICT"
    assert np.isnan(res["label_primary"])


def test_v2_unmatched_dxsum_row_no_conflict(base_config):
    """v2: Unmatched DXSUM visit dates (no corresponding PHC row) do NOT trigger conflict exclusion."""
    idx_date = pd.Timestamp("2020-01-01")
    fu_date = idx_date + pd.Timedelta(days=700)
    phc_sub = pd.DataFrame([
        {"RID": 207, "EXAMDATE": idx_date, "VISCODE2": "bl", "PHC_Diagnosis": 2},
        {"RID": 207, "EXAMDATE": fu_date, "VISCODE2": "m24", "PHC_Diagnosis": 2},
    ])
    # DXSUM has matching row at idx_date, but an unmatched row at day 100 with dx_code 1
    # Since PHC has no exam on day 100, Amendment 1e specifies unmatched rows are reported separately and do not exclude
    dxsum_sub = pd.DataFrame([
        {"RID": 207, "EXAMDATE": idx_date, "dx_code": 2},
        {"RID": 207, "EXAMDATE": idx_date + pd.Timedelta(days=100), "dx_code": 1},
        {"RID": 207, "EXAMDATE": fu_date, "dx_code": 2},
    ])

    res = evaluate_participant_timeline_v2(
        rid=207,
        idx_date=idx_date,
        phc_sub=phc_sub,
        dxsum_sub=dxsum_sub,
        clin_dates=[idx_date],
        fs_dates=[idx_date],
        mri_dates=[idx_date],
        c_cfg=base_config,
    )

    assert res["conflict_flag"] is False
    assert res["exclusion_reason"] == "ELIGIBLE"
    assert res["label_primary"] == 0


def test_v2_ad_on_or_before_predictor_cutoff(base_config):
    """v2: AD diagnosed on or before predictor cutoff date excludes participant."""
    idx_date = pd.Timestamp("2020-01-01")
    # Scan at day 30 post-anchor -> predictor cutoff is day 30
    scan_date = idx_date + pd.Timedelta(days=30)
    # AD diagnosed on day 15 (before scan date)
    phc_sub = pd.DataFrame([
        {"RID": 208, "EXAMDATE": idx_date, "VISCODE2": "bl", "PHC_Diagnosis": 2},
        {"RID": 208, "EXAMDATE": idx_date + pd.Timedelta(days=15), "VISCODE2": "m01", "PHC_Diagnosis": 3},
    ])
    dxsum_sub = pd.DataFrame([
        {"RID": 208, "EXAMDATE": idx_date, "dx_code": 2},
    ])

    res = evaluate_participant_timeline_v2(
        rid=208,
        idx_date=idx_date,
        phc_sub=phc_sub,
        dxsum_sub=dxsum_sub,
        clin_dates=[idx_date],
        fs_dates=[scan_date],
        mri_dates=[scan_date],
        c_cfg=base_config,
    )

    assert res["predictor_cutoff_date"] == scan_date.strftime("%Y-%m-%d")
    assert res["scan_after_index_visit"] is True
    assert res["exclusion_reason"] == "AD_ON_OR_BEFORE_PREDICTOR_CUTOFF"
    assert np.isnan(res["label_primary"])


def test_v2_exact_boundary_day_639_vs_640(base_config):
    """v2: Visit at day 639 fails primary negative (< 640), but visit at day 640 qualifies."""
    idx_date = pd.Timestamp("2020-01-01")

    # Participant A: Visit at day 639
    phc_639 = pd.DataFrame([
        {"RID": 209, "EXAMDATE": idx_date, "VISCODE2": "bl", "PHC_Diagnosis": 2},
        {"RID": 209, "EXAMDATE": idx_date + pd.Timedelta(days=639), "VISCODE2": "m21", "PHC_Diagnosis": 2},
    ])
    dxsum_639 = pd.DataFrame([{"RID": 209, "EXAMDATE": idx_date, "dx_code": 2}])

    res_639 = evaluate_participant_timeline_v2(
        rid=209,
        idx_date=idx_date,
        phc_sub=phc_639,
        dxsum_sub=dxsum_639,
        clin_dates=[idx_date],
        fs_dates=[idx_date],
        mri_dates=[idx_date],
        c_cfg=base_config,
    )
    assert res_639["exclusion_reason"] == "INSUFFICIENT_FOLLOWUP"
    assert np.isnan(res_639["label_primary"])
    # Day 639 qualifies for sens_ge550 and sens_symm180
    assert res_639["label_sens_550"] == 0
    assert res_639["label_sens_symm180"] == 0

    # Participant B: Visit at day 640
    phc_640 = pd.DataFrame([
        {"RID": 210, "EXAMDATE": idx_date, "VISCODE2": "bl", "PHC_Diagnosis": 2},
        {"RID": 210, "EXAMDATE": idx_date + pd.Timedelta(days=640), "VISCODE2": "m21", "PHC_Diagnosis": 2},
    ])
    dxsum_640 = pd.DataFrame([{"RID": 210, "EXAMDATE": idx_date, "dx_code": 2}])

    res_640 = evaluate_participant_timeline_v2(
        rid=210,
        idx_date=idx_date,
        phc_sub=phc_640,
        dxsum_sub=dxsum_640,
        clin_dates=[idx_date],
        fs_dates=[idx_date],
        mri_dates=[idx_date],
        c_cfg=base_config,
    )
    assert res_640["exclusion_reason"] == "ELIGIBLE"
    assert res_640["label_primary"] == 0
    assert res_640["label_sens_symm90"] == 0
    assert np.isnan(res_640["label_sens_ge730"])


def test_v2_exact_boundary_day_730_and_820(base_config):
    """v2: Tests exactly at day 730 (primary positive and negative) and day 820."""
    idx_date = pd.Timestamp("2020-01-01")

    # AD conversion at day 730 -> primary positive
    phc_ad730 = pd.DataFrame([
        {"RID": 211, "EXAMDATE": idx_date, "VISCODE2": "bl", "PHC_Diagnosis": 2},
        {"RID": 211, "EXAMDATE": idx_date + pd.Timedelta(days=730), "VISCODE2": "m24", "PHC_Diagnosis": 3},
    ])
    res_ad730 = evaluate_participant_timeline_v2(
        rid=211,
        idx_date=idx_date,
        phc_sub=phc_ad730,
        dxsum_sub=pd.DataFrame([{"RID": 211, "EXAMDATE": idx_date, "dx_code": 2}]),
        clin_dates=[idx_date],
        fs_dates=[idx_date],
        mri_dates=[idx_date],
        c_cfg=base_config,
    )
    assert res_ad730["label_primary"] == 1
    assert res_ad730["exclusion_reason"] == "ELIGIBLE"

    # Negative visit at day 730 -> qualifies under all negative definitions
    phc_neg730 = pd.DataFrame([
        {"RID": 212, "EXAMDATE": idx_date, "VISCODE2": "bl", "PHC_Diagnosis": 2},
        {"RID": 212, "EXAMDATE": idx_date + pd.Timedelta(days=730), "VISCODE2": "m24", "PHC_Diagnosis": 2},
    ])
    res_neg730 = evaluate_participant_timeline_v2(
        rid=212,
        idx_date=idx_date,
        phc_sub=phc_neg730,
        dxsum_sub=pd.DataFrame([{"RID": 212, "EXAMDATE": idx_date, "dx_code": 2}]),
        clin_dates=[idx_date],
        fs_dates=[idx_date],
        mri_dates=[idx_date],
        c_cfg=base_config,
    )
    assert res_neg730["label_primary"] == 0
    assert res_neg730["label_sens_ge730"] == 0

    # AD conversion at day 820 -> sensitivity positive, not primary positive
    phc_ad820 = pd.DataFrame([
        {"RID": 213, "EXAMDATE": idx_date, "VISCODE2": "bl", "PHC_Diagnosis": 2},
        {"RID": 213, "EXAMDATE": idx_date + pd.Timedelta(days=820), "VISCODE2": "m27", "PHC_Diagnosis": 3},
    ])
    res_ad820 = evaluate_participant_timeline_v2(
        rid=213,
        idx_date=idx_date,
        phc_sub=phc_ad820,
        dxsum_sub=pd.DataFrame([{"RID": 213, "EXAMDATE": idx_date, "dx_code": 2}]),
        clin_dates=[idx_date],
        fs_dates=[idx_date],
        mri_dates=[idx_date],
        c_cfg=base_config,
    )
    assert np.isnan(res_ad820["label_primary"])
    assert res_ad820["label_sens_ad820"] == 1


def test_v2_scan_boundaries_and_assertions(base_config):
    """v2: Scans at exactly anchor - 180 d and anchor + 90 d pass; > +90 d raises AssertionError."""
    idx_date = pd.Timestamp("2020-01-01")
    scan_m180 = idx_date - pd.Timedelta(days=180)
    scan_p90 = idx_date + pd.Timedelta(days=90)

    # Valid boundary scans
    res = evaluate_participant_timeline_v2(
        rid=214,
        idx_date=idx_date,
        phc_sub=pd.DataFrame([
            {"RID": 214, "EXAMDATE": idx_date, "VISCODE2": "bl", "PHC_Diagnosis": 2},
            {"RID": 214, "EXAMDATE": idx_date + pd.Timedelta(days=700), "VISCODE2": "m24", "PHC_Diagnosis": 2},
        ]),
        dxsum_sub=pd.DataFrame([{"RID": 214, "EXAMDATE": idx_date, "dx_code": 2}]),
        clin_dates=[idx_date],
        fs_dates=[scan_m180],
        mri_dates=[scan_p90],
        c_cfg=base_config,
    )
    assert res["predictor_cutoff_date"] == scan_p90.strftime("%Y-%m-%d")
    assert res["scan_after_index_visit"] is True

    # Out of boundary scan (> +90 d) must raise AssertionError
    scan_p91 = idx_date + pd.Timedelta(days=91)
    with pytest.raises(AssertionError, match="exceeds T_anchor \\+ 90 d"):
        evaluate_participant_timeline_v2(
            rid=215,
            idx_date=idx_date,
            phc_sub=pd.DataFrame([{"RID": 215, "EXAMDATE": idx_date, "PHC_Diagnosis": 2}]),
            dxsum_sub=pd.DataFrame(),
            clin_dates=[idx_date],
            fs_dates=[scan_p91],
            mri_dates=[],
            c_cfg=base_config,
        )


def test_v2_missing_core_clinical_baseline(base_config):
    """v2: Participant lacking core clinical set at index visit is excluded under MISSING_CORE_CLINICAL_BASELINE."""
    idx_date = pd.Timestamp("2020-01-01")
    res = evaluate_participant_timeline_v2(
        rid=216,
        idx_date=idx_date,
        phc_sub=pd.DataFrame([
            {"RID": 216, "EXAMDATE": idx_date, "VISCODE2": "bl", "PHC_Diagnosis": 2},
            {"RID": 216, "EXAMDATE": idx_date + pd.Timedelta(days=700), "VISCODE2": "m24", "PHC_Diagnosis": 2},
        ]),
        dxsum_sub=pd.DataFrame([{"RID": 216, "EXAMDATE": idx_date, "dx_code": 2}]),
        clin_dates=[],
        fs_dates=[idx_date],
        mri_dates=[idx_date],
        has_core_clinical=False,
        c_cfg=base_config,
    )
    assert res["exclusion_reason"] == "MISSING_CORE_CLINICAL_BASELINE"
    assert np.isnan(res["label_primary"])
