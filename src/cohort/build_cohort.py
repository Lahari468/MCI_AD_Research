#!/usr/bin/env python3
"""Config-driven Cohort Construction Pipeline for MCI to AD Progression Research.
Supports:
  - Version 1 (Stage 3A): Produces data/processed/cohort_v1.csv
  - Version 2 (Stage 3A-2): Produces data/processed/cohort_v2.csv
Adheres strictly to approved decision logs (decisions_log.md and decisions_log_v2.md).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def load_config(config_path: Path | str = "configs/base.yaml") -> dict[str, Any]:
    """Load configuration YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def recode_negative_values(df: pd.DataFrame, col: str) -> tuple[pd.Series, pd.Series]:
    """Recode values < 0 in clinical metrics to NaN and generate an audit flag."""
    if col not in df.columns:
        return pd.Series(dtype=float), pd.Series(dtype=bool)

    numeric_vals = pd.to_numeric(df[col], errors="coerce")
    is_neg = (numeric_vals < 0) & numeric_vals.notna()
    cleaned = numeric_vals.mask(is_neg, np.nan)
    return cleaned, is_neg


def parse_rid_from_ptid(ptid_val: Any) -> int | None:
    """Extract numeric RID from PTID string (e.g. '011_S_0010' -> 10)."""
    if pd.isna(ptid_val):
        return None
    s = str(ptid_val).strip()
    parts = s.split("_")
    try:
        return int(parts[-1])
    except (ValueError, IndexError):
        return None


def derive_adni_phase(prot_str: Any) -> str:
    """Standardize ADNI phase string."""
    if pd.isna(prot_str):
        return "Unknown"
    s = str(prot_str).upper()
    for phase in ["ADNI1", "ADNIGO", "ADNI2", "ADNI3", "ADNI4"]:
        if phase in s:
            return phase
    return s


def evaluate_participant_timeline(
    rid: int,
    idx_date: pd.Timestamp,
    phc_sub: pd.DataFrame,
    dxsum_sub: pd.DataFrame,
    clin_dates: list[pd.Timestamp],
    fs_dates: list[pd.Timestamp],
    mri_dates: list[pd.Timestamp],
    idx_visit_code: str = "bl",
    ptid: str | None = None,
    adni_phase: str = "Unknown",
    c_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a single participant's timeline under Cohort v1 specifications."""
    if c_cfg is None:
        c_cfg = {}

    horizon = c_cfg.get("prediction_horizon_days", 730)
    min_neg_days = c_cfg.get("primary_negative_min_days", 640)
    sens_pos_max = c_cfg.get("sens_pos_max_days", 820)
    sens_ge550 = c_cfg.get("sens_neg_ge550_days", 550)
    sens_symm90_min = c_cfg.get("sens_neg_symm90_min_days", 640)
    sens_symm90_max = c_cfg.get("sens_neg_symm90_max_days", 820)
    sens_symm180_min = c_cfg.get("sens_neg_symm180_min_days", 550)
    sens_symm180_max = c_cfg.get("sens_neg_symm180_max_days", 910)
    sens_ge730 = c_cfg.get("sens_neg_ge730_days", 730)

    all_used_dates = [d for d in (clin_dates + fs_dates + mri_dates) if pd.notna(d)]
    t0 = max([idx_date] + all_used_dates) if all_used_dates else idx_date

    imaging_after_idx = [d for d in (fs_dates + mri_dates) if pd.notna(d) and d > idx_date]
    scan_after_idx_flag = len(imaging_after_idx) > 0

    ad_on_or_before_t0 = any((phc_sub["EXAMDATE"] <= t0) & (phc_sub["PHC_Diagnosis"] == 3))
    fu_visits = phc_sub[phc_sub["EXAMDATE"] > t0].sort_values("EXAMDATE").copy()
    days_of_fu = (phc_sub["EXAMDATE"].max() - t0).days if not phc_sub.empty else 0

    non_ad_flag = any(phc_sub["PHC_Diagnosis"] == 4)
    reverter_flag = any((phc_sub["EXAMDATE"] > idx_date) & (phc_sub["PHC_Diagnosis"] == 1))

    conflict_flag = False
    source_visits = [f"index:{idx_visit_code}({idx_date.strftime('%Y-%m-%d')})"]

    dx_idx = dxsum_sub[dxsum_sub["EXAMDATE"] == idx_date]
    if not dx_idx.empty and not dx_idx["dx_code"].isna().all():
        if not (dx_idx["dx_code"] == 2).any():
            conflict_flag = True

    ad_events = fu_visits[(fu_visits["EXAMDATE"] <= t0 + pd.Timedelta(days=horizon)) & (fu_visits["PHC_Diagnosis"] == 3)]
    valid_neg_fu = fu_visits[fu_visits["PHC_Diagnosis"].isin([1, 2])].copy()
    if not valid_neg_fu.empty:
        valid_neg_fu["days_after_t0"] = (valid_neg_fu["EXAMDATE"] - t0).dt.days
    else:
        valid_neg_fu["days_after_t0"] = pd.Series(dtype=int)

    is_primary_positive = False
    is_primary_negative = False

    if not ad_events.empty:
        first_ad_row = ad_events.iloc[0]
        first_ad_date = first_ad_row["EXAMDATE"]
        first_ad_visit = first_ad_row.get("VISCODE2", "fu")
        is_primary_positive = True
        source_visits.append(f"AD_event:{first_ad_visit}({first_ad_date.strftime('%Y-%m-%d')})")

        dx_ad = dxsum_sub[dxsum_sub["EXAMDATE"] == first_ad_date]
        if not dx_ad.empty and not dx_ad["dx_code"].isna().all():
            if not (dx_ad["dx_code"] == 3).any():
                conflict_flag = True
    else:
        qual_neg = valid_neg_fu[valid_neg_fu["days_after_t0"] >= min_neg_days]
        if not qual_neg.empty:
            is_primary_negative = True
            qual_row = qual_neg.iloc[0]
            qual_visit = qual_row.get("VISCODE2", "fu")
            source_visits.append(f"qual_neg:{qual_visit}({qual_row['EXAMDATE'].strftime('%Y-%m-%d')})")

            dx_neg = dxsum_sub[dxsum_sub["EXAMDATE"] == qual_row["EXAMDATE"]]
            if not dx_neg.empty and not dx_neg["dx_code"].isna().all():
                if not (dx_neg["dx_code"].isin([1, 2])).any():
                    conflict_flag = True

    ad_820 = fu_visits[(fu_visits["EXAMDATE"] <= t0 + pd.Timedelta(days=sens_pos_max)) & (fu_visits["PHC_Diagnosis"] == 3)]
    label_sens_ad820 = 1 if not ad_820.empty else (0 if is_primary_negative else np.nan)

    label_sens_550 = np.nan
    label_sens_symm90 = np.nan
    label_sens_symm180 = np.nan
    label_sens_ge730 = np.nan

    if is_primary_positive:
        label_sens_550 = 1
        label_sens_symm90 = 1
        label_sens_symm180 = 1
        label_sens_ge730 = 1
    elif ad_events.empty:
        if not valid_neg_fu.empty:
            if (valid_neg_fu["days_after_t0"] >= sens_ge550).any():
                label_sens_550 = 0
            if ((valid_neg_fu["days_after_t0"] >= sens_symm90_min) & (valid_neg_fu["days_after_t0"] <= sens_symm90_max)).any():
                label_sens_symm90 = 0
            if ((valid_neg_fu["days_after_t0"] >= sens_symm180_min) & (valid_neg_fu["days_after_t0"] <= sens_symm180_max)).any():
                label_sens_symm180 = 0
            if (valid_neg_fu["days_after_t0"] >= sens_ge730).any():
                label_sens_ge730 = 0

    exclusion_reason = None
    if ad_on_or_before_t0:
        exclusion_reason = "AD_ON_OR_BEFORE_T0"
    elif non_ad_flag:
        exclusion_reason = "NON_AD_DEMENTIA"
    elif conflict_flag:
        exclusion_reason = "PHC_DXSUM_CONFLICT"
    elif not is_primary_positive and not is_primary_negative:
        exclusion_reason = "INSUFFICIENT_FOLLOWUP"

    label_primary = np.nan
    if exclusion_reason is None:
        if is_primary_positive:
            label_primary = 1
        elif is_primary_negative:
            label_primary = 0

    return {
        "RID": rid,
        "PTID": ptid if ptid else f"RID_{rid}",
        "index_visit_date": idx_date.strftime("%Y-%m-%d"),
        "index_visit_code": idx_visit_code,
        "T0": t0.strftime("%Y-%m-%d"),
        "scan_after_index_visit": scan_after_idx_flag,
        "label_primary": label_primary,
        "label_sens_550": label_sens_550,
        "label_sens_symm90": label_sens_symm90,
        "label_sens_symm180": label_sens_symm180,
        "label_sens_ge730": label_sens_ge730,
        "label_sens_ad820": label_sens_ad820,
        "exclusion_reason": exclusion_reason if exclusion_reason else "ELIGIBLE",
        "reverter_flag": reverter_flag,
        "non_ad_flag": non_ad_flag,
        "conflict_flag": conflict_flag,
        "adni_phase": adni_phase,
        "days_of_followup": days_of_fu,
        "source_visits_used": "; ".join(source_visits),
    }


def evaluate_participant_timeline_v2(
    rid: int,
    idx_date: pd.Timestamp,
    phc_sub: pd.DataFrame,
    dxsum_sub: pd.DataFrame,
    clin_dates: list[pd.Timestamp],
    fs_dates: list[pd.Timestamp],
    mri_dates: list[pd.Timestamp],
    idx_visit_code: str = "bl",
    ptid: str | None = None,
    adni_phase: str = "Unknown",
    entry_type: str = "enrolled_as_MCI",
    has_core_clinical: bool = True,
    c_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a single participant's timeline under Cohort v2 specifications (Amendments 1a-1j).

    - Label anchor T_anchor = index visit date (fixed).
    - Predictor cutoff date = latest date among predictors used.
    - Exclude if AD diagnosed on or before predictor cutoff date.
    - Exclude if PHC_Diagnosis in {4, 5} within (T_anchor, T_anchor + 910 d].
    - Cross-check PHC vs DXSUM on matched EXAMDATE in (T_anchor, T_anchor + 910 d] + index visit.
    """
    if c_cfg is None:
        c_cfg = {}

    horizon = c_cfg.get("prediction_horizon_days", 730)
    min_neg_days = c_cfg.get("primary_negative_min_days", 640)
    sens_pos_max = c_cfg.get("sens_pos_max_days", 820)
    sens_ge550 = c_cfg.get("sens_neg_ge550_days", 550)
    sens_symm90_min = c_cfg.get("sens_neg_symm90_min_days", 640)
    sens_symm90_max = c_cfg.get("sens_neg_symm90_max_days", 820)
    sens_symm180_min = c_cfg.get("sens_neg_symm180_min_days", 550)
    sens_symm180_max = c_cfg.get("sens_neg_symm180_max_days", 910)
    sens_ge730 = c_cfg.get("sens_neg_ge730_days", 730)
    crosscheck_horizon = c_cfg.get("crosscheck_horizon_days", 910)

    t_anchor = idx_date

    # Assert predictor window validity [T_anchor - 180 d, T_anchor + 90 d]
    for d in (fs_dates + mri_dates):
        if pd.notna(d):
            assert d <= t_anchor + pd.Timedelta(days=90), f"Predictor imaging date {d} exceeds T_anchor + 90 d"
            assert d >= t_anchor - pd.Timedelta(days=180), f"Predictor imaging date {d} precedes T_anchor - 180 d"
    for d in clin_dates:
        if pd.notna(d):
            assert d <= t_anchor, f"Clinical predictor date {d} exceeds T_anchor"
            assert d >= t_anchor - pd.Timedelta(days=90), f"Clinical predictor date {d} precedes T_anchor - 90 d"

    # Predictor cutoff date
    all_used_dates = [d for d in (clin_dates + fs_dates + mri_dates) if pd.notna(d)]
    predictor_cutoff_date = max([t_anchor] + all_used_dates) if all_used_dates else t_anchor

    imaging_after_idx = [d for d in (fs_dates + mri_dates) if pd.notna(d) and d > t_anchor]
    scan_after_idx_flag = len(imaging_after_idx) > 0

    # Clean PHC sub: missing diagnosis is treated as NaN, never non-progression
    phc_valid = phc_sub.sort_values("EXAMDATE").copy()

    # 1. AD on or before predictor cutoff
    ad_on_or_before_cutoff = any((phc_valid["EXAMDATE"] <= predictor_cutoff_date) & (phc_valid["PHC_Diagnosis"] == 3))

    # 2. Competing non-AD dementia or other cognitive diagnosis (codes 4 and 5) in (T_anchor, T_anchor + 910 d]
    competing_events = phc_valid[
        (phc_valid["EXAMDATE"] > t_anchor)
        & (phc_valid["EXAMDATE"] <= t_anchor + pd.Timedelta(days=crosscheck_horizon))
        & (phc_valid["PHC_Diagnosis"].isin([4, 5]))
    ]
    non_ad_or_other_flag = not competing_events.empty

    # 3. Reverter flag: any diagnosis == 1 after T_anchor
    reverter_flag = any((phc_valid["EXAMDATE"] > t_anchor) & (phc_valid["PHC_Diagnosis"] == 1))

    # 4. Conflict check on matched visits in [T_anchor, T_anchor + 910 d]
    eval_visits = phc_valid[
        (phc_valid["EXAMDATE"] >= t_anchor)
        & (phc_valid["EXAMDATE"] <= t_anchor + pd.Timedelta(days=crosscheck_horizon))
    ]
    conflict_flag = False
    source_visits = [f"index:{idx_visit_code}({t_anchor.strftime('%Y-%m-%d')})"]

    for _, v_row in eval_visits.iterrows():
        v_date = v_row["EXAMDATE"]
        p_dx = v_row["PHC_Diagnosis"]
        if pd.isna(p_dx):
            continue

        # Check matched DXSUM visit
        dx_match = dxsum_sub[dxsum_sub["EXAMDATE"] == v_date]
        if not dx_match.empty and not dx_match["dx_code"].isna().all():
            dx_codes = dx_match["dx_code"].dropna().unique().tolist()
            if dx_codes:
                # Compare: if PHC diagnosis in {1, 2, 3} and no dx_code matches
                if p_dx in [1, 2, 3] and not any(p_dx == dc for dc in dx_codes):
                    conflict_flag = True

    # Outcomes count strictly from T_anchor
    fu_visits = phc_valid[phc_valid["EXAMDATE"] > t_anchor].copy()
    days_of_fu = (phc_valid["EXAMDATE"].max() - t_anchor).days if not phc_valid.empty else 0

    # Positive: first PHC_Diagnosis == 3 in (T_anchor, T_anchor + 730 d]
    ad_events = fu_visits[(fu_visits["EXAMDATE"] <= t_anchor + pd.Timedelta(days=horizon)) & (fu_visits["PHC_Diagnosis"] == 3)]

    # Negative: visit with diagnosis in {1, 2} at day >= 640 and no diagnosis == 3 <= 730 d
    valid_neg_fu = fu_visits[fu_visits["PHC_Diagnosis"].isin([1, 2])].copy()
    if not valid_neg_fu.empty:
        valid_neg_fu["days_after_anchor"] = (valid_neg_fu["EXAMDATE"] - t_anchor).dt.days
    else:
        valid_neg_fu["days_after_anchor"] = pd.Series(dtype=int)

    is_primary_positive = False
    is_primary_negative = False

    if not ad_events.empty:
        first_ad_row = ad_events.iloc[0]
        first_ad_date = first_ad_row["EXAMDATE"]
        first_ad_visit = first_ad_row.get("VISCODE2", "fu")
        is_primary_positive = True
        source_visits.append(f"AD_event:{first_ad_visit}({first_ad_date.strftime('%Y-%m-%d')})")
    else:
        qual_neg = valid_neg_fu[valid_neg_fu["days_after_anchor"] >= min_neg_days]
        if not qual_neg.empty:
            is_primary_negative = True
            qual_row = qual_neg.iloc[0]
            qual_visit = qual_row.get("VISCODE2", "fu")
            source_visits.append(f"qual_neg:{qual_visit}({qual_row['EXAMDATE'].strftime('%Y-%m-%d')})")

    # Sensitivity labels
    ad_820 = fu_visits[(fu_visits["EXAMDATE"] <= t_anchor + pd.Timedelta(days=sens_pos_max)) & (fu_visits["PHC_Diagnosis"] == 3)]
    label_sens_ad820 = 1 if not ad_820.empty else (0 if is_primary_negative else np.nan)

    label_sens_550 = np.nan
    label_sens_symm90 = np.nan
    label_sens_symm180 = np.nan
    label_sens_ge730 = np.nan

    if is_primary_positive:
        label_sens_550 = 1
        label_sens_symm90 = 1
        label_sens_symm180 = 1
        label_sens_ge730 = 1
    elif ad_events.empty:
        if not valid_neg_fu.empty:
            if (valid_neg_fu["days_after_anchor"] >= sens_ge550).any():
                label_sens_550 = 0
            if ((valid_neg_fu["days_after_anchor"] >= sens_symm90_min) & (valid_neg_fu["days_after_anchor"] <= sens_symm90_max)).any():
                label_sens_symm90 = 0
            if ((valid_neg_fu["days_after_anchor"] >= sens_symm180_min) & (valid_neg_fu["days_after_anchor"] <= sens_symm180_max)).any():
                label_sens_symm180 = 0
            if (valid_neg_fu["days_after_anchor"] >= sens_ge730).any():
                label_sens_ge730 = 0

    # Strict sensitivity variant: predictors dated <= T_anchor
    # If participant was excluded due to AD on or before predictor cutoff because of a post-anchor scan,
    # under strict pre-anchor they would be evaluated using cutoff = T_anchor.
    label_sens_strict_pre_anchor = np.nan
    ad_before_t_anchor = any((phc_valid["EXAMDATE"] <= t_anchor) & (phc_valid["PHC_Diagnosis"] == 3))
    if has_core_clinical and not non_ad_or_other_flag and not conflict_flag and not ad_before_t_anchor:
        if is_primary_positive:
            label_sens_strict_pre_anchor = 1
        elif is_primary_negative:
            label_sens_strict_pre_anchor = 0

    # Hierarchical exclusions (mutually exclusive)
    exclusion_reason = None
    if not has_core_clinical:
        exclusion_reason = "MISSING_CORE_CLINICAL_BASELINE"
    elif ad_on_or_before_cutoff:
        exclusion_reason = "AD_ON_OR_BEFORE_PREDICTOR_CUTOFF"
    elif non_ad_or_other_flag:
        exclusion_reason = "NON_AD_OR_OTHER_DEMENTIA"
    elif conflict_flag:
        exclusion_reason = "PHC_DXSUM_CONFLICT"
    elif not is_primary_positive and not is_primary_negative:
        exclusion_reason = "INSUFFICIENT_FOLLOWUP"

    label_primary = np.nan
    if exclusion_reason is None:
        label_primary = 1 if is_primary_positive else 0

    return {
        "RID": rid,
        "PTID": ptid if ptid else f"RID_{rid}",
        "index_visit_date": t_anchor.strftime("%Y-%m-%d"),
        "index_visit_code": idx_visit_code,
        "T_anchor": t_anchor.strftime("%Y-%m-%d"),
        "entry_type": entry_type,
        "adni_phase": adni_phase,
        "predictor_cutoff_date": predictor_cutoff_date.strftime("%Y-%m-%d"),
        "scan_after_index_visit": scan_after_idx_flag,
        "label_primary": label_primary,
        "label_sens_550": label_sens_550,
        "label_sens_symm90": label_sens_symm90,
        "label_sens_symm180": label_sens_symm180,
        "label_sens_ge730": label_sens_ge730,
        "label_sens_ad820": label_sens_ad820,
        "label_sens_strict_pre_anchor": label_sens_strict_pre_anchor,
        "exclusion_reason": exclusion_reason if exclusion_reason else "ELIGIBLE",
        "reverter_flag": reverter_flag,
        "non_ad_or_other_flag": non_ad_or_other_flag,
        "conflict_flag": conflict_flag,
        "days_of_followup": days_of_fu,
        "source_visits_used": "; ".join(source_visits),
    }


def build_cohort_v1(config_path: Path | str = "configs/base.yaml", out_path: str = "data/processed/cohort_v1.csv") -> pd.DataFrame:
    """Build standardized analysis cohort v1 (preserves existing v1 logic)."""
    cfg = load_config(config_path)
    c_cfg = cfg.get("cohort", {})

    print(f"Loading datasets for Cohort v1 using config: {config_path}...")
    phc_path = Path("data/interim/ADSP_PHC_DEMODX.csv")
    dxsum_path = Path("01_raw_metadata/DXSUM_26Oct2026.csv")
    prov_path = Path("06_audit_outputs/authoritative_provisional_baseline_MCI_cohort.csv")

    phc = pd.read_csv(phc_path, low_memory=False)
    phc["EXAMDATE"] = pd.to_datetime(phc["EXAMDATE"])
    phc["RID"] = phc["RID"].astype(int)

    dxsum = pd.read_csv(dxsum_path, low_memory=False)
    dxsum["EXAMDATE"] = pd.to_datetime(dxsum["EXAMDATE"])
    dxsum["RID"] = dxsum["RID"].astype(int)
    dxsum["dx_code"] = dxsum["DIAGNOSIS"].map({1.0: 1, 2.0: 2, 3.0: 3})

    prov = pd.read_csv(prov_path, low_memory=False)
    prov["baseline_date"] = pd.to_datetime(prov["baseline_date"])
    prov["RID"] = prov["RID"].astype(int)

    mmse = pd.read_csv("02_clinical/MMSE_26Oct2026.csv", low_memory=False)
    mmse["VISDATE"] = pd.to_datetime(mmse["VISDATE"])
    mmse["RID"] = mmse["RID"].astype(int)

    cdr = pd.read_csv("02_clinical/CDR_26Oct2026.csv", low_memory=False)
    cdr["VISDATE"] = pd.to_datetime(cdr["VISDATE"])
    cdr["RID"] = cdr["RID"].astype(int)

    faq = pd.read_csv("02_clinical/FAQ_26Oct2026.csv", low_memory=False)
    faq["VISDATE"] = pd.to_datetime(faq["VISDATE"])
    faq["RID"] = faq["RID"].astype(int)

    adas = pd.read_csv("02_clinical/ADAS_26Oct2026.csv", low_memory=False)
    adas["VISDATE"] = pd.to_datetime(adas["VISDATE"])
    adas["RID"] = adas["RID"].astype(int)

    ptdemog = pd.read_csv("02_clinical/PTDEMOG_26Oct2026.csv", low_memory=False)
    ptdemog["VISDATE"] = pd.to_datetime(ptdemog["VISDATE"])
    ptdemog["RID"] = ptdemog["RID"].astype(int)

    fs = pd.read_csv("03_freesurfer/UCSFFSX7_26Oct2026.csv", low_memory=False)
    fs["EXAMDATE"] = pd.to_datetime(fs["EXAMDATE"])
    fs["RID"] = fs["RID"].astype(int)

    mri_man = pd.read_csv("04_mri_metadata/T1w_Structural_MRI_Cohort_Manifest_26Oct2026.csv", low_memory=False)
    mri_man["image_date"] = pd.to_datetime(mri_man["image_date"])
    mri_man["RID"] = mri_man["subject_id"].apply(parse_rid_from_ptid)

    clin_pre = c_cfg.get("clinical_window_pre_days", 90)
    img_pre = c_cfg.get("imaging_window_pre_days", 180)
    img_post = c_cfg.get("imaging_window_post_days", 90)

    cohort_records = []
    for _, row in prov.iterrows():
        rid = int(row["RID"])
        idx_date = row["baseline_date"]

        phc_sub = phc[phc["RID"] == rid].sort_values("EXAMDATE")
        idx_match = phc_sub[phc_sub["EXAMDATE"] == idx_date]

        idx_visit_code = idx_match.iloc[0]["VISCODE2"] if not idx_match.empty else "bl"
        ptid = idx_match.iloc[0]["PTID"] if not idx_match.empty else f"RID_{rid}"
        adni_phase = derive_adni_phase(idx_match.iloc[0].get("ORIGPROT", idx_match.iloc[0].get("COLPROT", "Unknown"))) if not idx_match.empty else "Unknown"

        c_start = idx_date - pd.Timedelta(days=clin_pre)
        c_end = idx_date
        i_start = idx_date - pd.Timedelta(days=img_pre)
        i_end = idx_date + pd.Timedelta(days=img_post)

        clin_dates_used = []
        for df_c in [mmse, cdr, faq, adas, ptdemog]:
            s_c = df_c[(df_c["RID"] == rid) & (df_c["VISDATE"] >= c_start) & (df_c["VISDATE"] <= c_end)]
            clin_dates_used.extend(s_c["VISDATE"].dropna().tolist())

        fs_sub = fs[(fs["RID"] == rid) & (fs["EXAMDATE"] >= i_start) & (fs["EXAMDATE"] <= i_end)]
        fs_dates_used = fs_sub["EXAMDATE"].dropna().tolist()

        mri_sub = mri_man[(mri_man["RID"] == rid) & (mri_man["image_date"] >= i_start) & (mri_man["image_date"] <= i_end)]
        mri_dates_used = mri_sub["image_date"].dropna().tolist()

        rec = evaluate_participant_timeline(
            rid=rid,
            idx_date=idx_date,
            phc_sub=phc_sub,
            dxsum_sub=dxsum[dxsum["RID"] == rid],
            clin_dates=clin_dates_used,
            fs_dates=fs_dates_used,
            mri_dates=mri_dates_used,
            idx_visit_code=idx_visit_code,
            ptid=ptid,
            adni_phase=adni_phase,
            c_cfg=c_cfg,
        )
        cohort_records.append(rec)

    cohort_df = pd.DataFrame(cohort_records)
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cohort_df.to_csv(out_file, index=False)
    print(f"Cohort v1 saved to {out_file} ({len(cohort_df)} records).")
    return cohort_df


def build_cohort_v2(config_path: Path | str = "configs/base.yaml", out_path: str = "data/processed/cohort_v2.csv") -> pd.DataFrame:
    """Build standardized analysis cohort v2 adhering strictly to Stage 3A-2 amendments.

    Index visit = earliest PHC_Diagnosis==2 visit with complete core clinical set (MMSE, CDR, ADAS, FAQ)
    in [visit - 90 d, visit].
    T_anchor = index visit date (fixed per RID).
    Predictor window FreeSurfer/MRI = [T_anchor - 180 d, T_anchor + 90 d].
    """
    cfg = load_config(config_path)
    c_cfg = cfg.get("cohort", {})

    print(f"Loading datasets for Cohort v2 using config: {config_path}...")
    phc_path = Path("data/interim/ADSP_PHC_DEMODX.csv")
    dxsum_path = Path("01_raw_metadata/DXSUM_26Oct2026.csv")
    prov_path = Path("06_audit_outputs/authoritative_provisional_baseline_MCI_cohort.csv")

    phc = pd.read_csv(phc_path, low_memory=False)
    phc["EXAMDATE"] = pd.to_datetime(phc["EXAMDATE"])
    phc["RID"] = phc["RID"].astype(int)

    dxsum = pd.read_csv(dxsum_path, low_memory=False)
    dxsum["EXAMDATE"] = pd.to_datetime(dxsum["EXAMDATE"])
    dxsum["RID"] = dxsum["RID"].astype(int)
    dxsum["dx_code"] = dxsum["DIAGNOSIS"].map({1.0: 1, 2.0: 2, 3.0: 3})

    prov = pd.read_csv(prov_path, low_memory=False)
    prov["baseline_date"] = pd.to_datetime(prov["baseline_date"])
    prov["RID"] = prov["RID"].astype(int)

    # Clinical metrics cleaned of negative sentinel values (< 0 to NaN)
    mmse = pd.read_csv("02_clinical/MMSE_26Oct2026.csv", low_memory=False)
    mmse["VISDATE"] = pd.to_datetime(mmse["VISDATE"])
    mmse["RID"] = mmse["RID"].astype(int)
    mmse["score_clean"], mmse["was_neg"] = recode_negative_values(mmse, "MMSCORE")
    mmse_valid = mmse[mmse["score_clean"].notna()]

    cdr = pd.read_csv("02_clinical/CDR_26Oct2026.csv", low_memory=False)
    cdr["VISDATE"] = pd.to_datetime(cdr["VISDATE"])
    cdr["RID"] = cdr["RID"].astype(int)
    cdr["score_clean"], cdr["was_neg"] = recode_negative_values(cdr, "CDGLOBAL")
    cdr_valid = cdr[cdr["score_clean"].notna()]

    faq = pd.read_csv("02_clinical/FAQ_26Oct2026.csv", low_memory=False)
    faq["VISDATE"] = pd.to_datetime(faq["VISDATE"])
    faq["RID"] = faq["RID"].astype(int)
    faq["score_clean"], faq["was_neg"] = recode_negative_values(faq, "FAQTOTAL")
    faq_valid = faq[faq["score_clean"].notna()]

    adas = pd.read_csv("02_clinical/ADAS_26Oct2026.csv", low_memory=False)
    adas["VISDATE"] = pd.to_datetime(adas["VISDATE"])
    adas["RID"] = adas["RID"].astype(int)
    adas["tot_clean"], adas["tot_neg"] = recode_negative_values(adas, "TOTSCORE")
    adas["t13_clean"], adas["t13_neg"] = recode_negative_values(adas, "TOTAL13")
    adas_valid = adas[adas["tot_clean"].notna() | adas["t13_clean"].notna()]

    ptdemog = pd.read_csv("02_clinical/PTDEMOG_26Oct2026.csv", low_memory=False)
    ptdemog["VISDATE"] = pd.to_datetime(ptdemog["VISDATE"])
    ptdemog["RID"] = ptdemog["RID"].astype(int)

    fs = pd.read_csv("03_freesurfer/UCSFFSX7_26Oct2026.csv", low_memory=False)
    fs["EXAMDATE"] = pd.to_datetime(fs["EXAMDATE"])
    fs["RID"] = fs["RID"].astype(int)

    mri_man = pd.read_csv("04_mri_metadata/T1w_Structural_MRI_Cohort_Manifest_26Oct2026.csv", low_memory=False)
    mri_man["image_date"] = pd.to_datetime(mri_man["image_date"])
    mri_man["RID"] = mri_man["subject_id"].apply(parse_rid_from_ptid)

    clin_pre = c_cfg.get("clinical_window_pre_days", 90)
    img_pre = c_cfg.get("imaging_window_pre_days", 180)
    img_post = c_cfg.get("imaging_window_post_days", 90)

    cohort_records = []
    print(f"Building Cohort v2 records for {len(prov)} baseline MCI participants...")

    for _, row in prov.iterrows():
        rid = int(row["RID"])
        sub_phc = phc[phc["RID"] == rid].sort_values("EXAMDATE")
        sub_mci = sub_phc[sub_phc["PHC_Diagnosis"] == 2]

        # 1. Index visit: earliest PHC_Diagnosis==2 visit with core clinical set in [visit-90d, visit]
        chosen_idx = None
        for _, mci_row in sub_mci.iterrows():
            d = mci_row["EXAMDATE"]
            w_start = d - pd.Timedelta(days=clin_pre)
            has_m = not mmse_valid[(mmse_valid["RID"] == rid) & (mmse_valid["VISDATE"] >= w_start) & (mmse_valid["VISDATE"] <= d)].empty
            has_c = not cdr_valid[(cdr_valid["RID"] == rid) & (cdr_valid["VISDATE"] >= w_start) & (cdr_valid["VISDATE"] <= d)].empty
            has_f = not faq_valid[(faq_valid["RID"] == rid) & (faq_valid["VISDATE"] >= w_start) & (faq_valid["VISDATE"] <= d)].empty
            has_a = not adas_valid[(adas_valid["RID"] == rid) & (adas_valid["VISDATE"] >= w_start) & (adas_valid["VISDATE"] <= d)].empty

            if has_m and has_c and has_f and has_a:
                chosen_idx = mci_row
                break

        if chosen_idx is not None:
            has_core = True
            idx_date = chosen_idx["EXAMDATE"]
            idx_visit_code = chosen_idx.get("VISCODE2", "bl")
            ptid = chosen_idx.get("PTID", f"RID_{rid}")
            raw_prot = chosen_idx.get("ORIGPROT", chosen_idx.get("COLPROT", "Unknown"))
            adni_phase = derive_adni_phase(raw_prot)
        else:
            has_core = False
            # Fallback to earliest MCI visit for baseline timing and audit
            fallback_row = sub_mci.iloc[0] if not sub_mci.empty else sub_phc.iloc[0]
            idx_date = fallback_row["EXAMDATE"]
            idx_visit_code = fallback_row.get("VISCODE2", "bl")
            ptid = fallback_row.get("PTID", f"RID_{rid}")
            raw_prot = fallback_row.get("ORIGPROT", fallback_row.get("COLPROT", "Unknown"))
            adni_phase = derive_adni_phase(raw_prot)

        # entry_type: enrolled as MCI vs earlier CN or other visit exists
        prior_visits = sub_phc[sub_phc["EXAMDATE"] < idx_date]
        if prior_visits.empty:
            entry_type = "enrolled_as_MCI"
        else:
            entry_type = "progressed_from_CN"

        # Predictor collections
        c_start = idx_date - pd.Timedelta(days=clin_pre)
        c_end = idx_date
        i_start = idx_date - pd.Timedelta(days=img_pre)
        i_end = idx_date + pd.Timedelta(days=img_post)

        clin_dates_used = []
        for df_c in [mmse, cdr, faq, adas, ptdemog]:
            s_c = df_c[(df_c["RID"] == rid) & (df_c["VISDATE"] >= c_start) & (df_c["VISDATE"] <= c_end)]
            clin_dates_used.extend(s_c["VISDATE"].dropna().tolist())

        fs_sub = fs[(fs["RID"] == rid) & (fs["EXAMDATE"] >= i_start) & (fs["EXAMDATE"] <= i_end)]
        fs_dates_used = fs_sub["EXAMDATE"].dropna().tolist()

        mri_sub = mri_man[(mri_man["RID"] == rid) & (mri_man["image_date"] >= i_start) & (mri_man["image_date"] <= i_end)]
        mri_dates_used = mri_sub["image_date"].dropna().tolist()

        rec = evaluate_participant_timeline_v2(
            rid=rid,
            idx_date=idx_date,
            phc_sub=sub_phc,
            dxsum_sub=dxsum[dxsum["RID"] == rid],
            clin_dates=clin_dates_used,
            fs_dates=fs_dates_used,
            mri_dates=mri_dates_used,
            idx_visit_code=idx_visit_code,
            ptid=ptid,
            adni_phase=adni_phase,
            entry_type=entry_type,
            has_core_clinical=has_core,
            c_cfg=c_cfg,
        )
        cohort_records.append(rec)

    cohort_df = pd.DataFrame(cohort_records)
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cohort_df.to_csv(out_file, index=False)

    print(f"\nCohort v2 successfully constructed and saved to {out_file} ({len(cohort_df)} participants).")
    print("\nSummary of Exclusions:")
    print(cohort_df["exclusion_reason"].value_counts())
    print("\nPrimary Label Distribution (Eligible):")
    print(cohort_df["label_primary"].value_counts(dropna=False))
    print("\nEntry Type Breakdown:")
    print(cohort_df["entry_type"].value_counts())
    print(f"\nReverters (MCI -> CN post-anchor): {cohort_df['reverter_flag'].sum()}")
    print(f"Imaging after index visit: {cohort_df['scan_after_index_visit'].sum()}")
    print(f"Matched PHC vs DXSUM conflicts: {cohort_df['conflict_flag'].sum()}")

    return cohort_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build MCI to AD Research Cohort (v1 or v2)")
    parser.add_argument("--version", choices=["v1", "v2"], default="v2", help="Cohort specification version (default: v2)")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to base configuration YAML")
    parser.add_argument("--output", default=None, help="Optional output CSV path")
    args = parser.parse_args()

    if args.version == "v1":
        out = args.output if args.output else "data/processed/cohort_v1.csv"
        build_cohort_v1(config_path=args.config, out_path=out)
    else:
        out = args.output if args.output else "data/processed/cohort_v2.csv"
        build_cohort_v2(config_path=args.config, out_path=out)
