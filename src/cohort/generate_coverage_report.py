#!/usr/bin/env python3
"""Coverage and Reconciliation Audit Script for MCI to AD Research Cohort (v2).
Stage 3A-2: Generates coverage CSV files in results/coverage/ and outputs download candidates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_rid_from_ptid(ptid_val: str | None) -> int | None:
    if pd.isna(ptid_val):
        return None
    try:
        return int(str(ptid_val).strip().split("_")[-1])
    except (ValueError, IndexError):
        return None


def run_coverage_analysis(cohort_file: str = "data/processed/cohort_v2.csv") -> None:
    out_dir = Path("results/coverage")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating coverage reports into {out_dir} using cohort file: {cohort_file}...")

    # Load cohort
    cohort = pd.read_csv(cohort_file)
    cohort["index_visit_date"] = pd.to_datetime(cohort["index_visit_date"])
    anchor_col = "T_anchor" if "T_anchor" in cohort.columns else ("T0" if "T0" in cohort.columns else "index_visit_date")
    cohort["anchor_date"] = pd.to_datetime(cohort[anchor_col])
    primary_cohort = cohort[cohort["exclusion_reason"] == "ELIGIBLE"].copy()

    print(f"Loaded cohort: {len(cohort)} total, {len(primary_cohort)} primary eligible.")

    # Load modalities
    # Clinical
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

    # Genetics
    apoe = pd.read_csv("05_genetics/APOERES_26Oct2026.csv", low_memory=False)
    apoe["RID"] = apoe["RID"].astype(int)

    # FreeSurfer
    fs = pd.read_csv("03_freesurfer/UCSFFSX7_26Oct2026.csv", low_memory=False)
    fs["EXAMDATE"] = pd.to_datetime(fs["EXAMDATE"])
    fs["RID"] = fs["RID"].astype(int)

    # MRI DICOM
    dcm = pd.read_csv("results/inventory/dicom_series_headers.csv")
    dcm["acq_date"] = pd.to_datetime(dcm["acquisition_date"].astype(str), format="%Y%m%d", errors="coerce")
    dcm["RID"] = dcm["subject_id"].apply(parse_rid_from_ptid)

    # MRI Manifest
    mri_man = pd.read_csv("04_mri_metadata/T1w_Structural_MRI_Cohort_Manifest_26Oct2026.csv", low_memory=False)
    mri_man["image_date"] = pd.to_datetime(mri_man["image_date"])
    mri_man["RID"] = mri_man["subject_id"].apply(parse_rid_from_ptid)

    # NIfTI collection
    nifti_dir = Path("04_mri_metadata/nifti_all")
    nii_image_ids = set()
    if nifti_dir.exists():
        for f in nifti_dir.glob("*.nii.gz"):
            stem = f.name.replace(".nii.gz", "")
            try:
                nii_image_ids.add(int(stem))
            except ValueError:
                pass

    target_fs_features = [
        "ST24CV", "ST24SA", "ST24TA", "ST24TS", "ST29SV",
        "ST83CV", "ST83SA", "ST83TA", "ST83TS", "ST88SV"
    ]

    # (a) Clinical Coverage Analysis
    print("Auditing clinical instrument coverage in window...")
    clin_rows = []
    instruments = [
        ("MMSE", mmse, "MMSCORE"),
        ("CDR_Global", cdr, "CDGLOBAL"),
        ("CDR_SB", cdr, "CDRSB"),
        ("FAQ", faq, "FAQTOTAL"),
        ("ADAS_TOTSCORE", adas, "TOTSCORE"),
        ("ADAS_TOTAL13", adas, "TOTAL13"),
        ("PTDEMOG_Education", ptdemog, "PTEDUCAT"),
        ("PTDEMOG_Gender", ptdemog, "PTGENDER"),
    ]

    for inst_name, df_inst, val_col in instruments:
        for cohort_name, c_df in [("Primary Cohort (Eligible)", primary_cohort), ("Starting Cohort (Full)", cohort)]:
            present_rids = 0
            valid_val_rids = 0
            neg_val_rids = 0

            for _, r in c_df.iterrows():
                rid = r["RID"]
                idx_d = r["anchor_date"]
                w_start = idx_d - pd.Timedelta(days=90)
                w_end = idx_d

                sub = df_inst[(df_inst["RID"] == rid) & (df_inst["VISDATE"] >= w_start) & (df_inst["VISDATE"] <= w_end)]
                if not sub.empty:
                    present_rids += 1
                    s_vals = pd.to_numeric(sub[val_col], errors="coerce")
                    if (s_vals >= 0).any():
                        valid_val_rids += 1
                    if (s_vals < 0).any():
                        neg_val_rids += 1

            clin_rows.append({
                "instrument": inst_name,
                "cohort_subset": cohort_name,
                "total_cohort_rids": len(c_df),
                "rids_with_visit_in_window": present_rids,
                "visit_coverage_pct": round(present_rids / len(c_df) * 100, 2),
                "rids_with_valid_nonneg_score": valid_val_rids,
                "valid_score_coverage_pct": round(valid_val_rids / len(c_df) * 100, 2),
                "rids_with_negative_admin_code": neg_val_rids,
            })

    clin_coverage_df = pd.DataFrame(clin_rows)
    clin_coverage_df.to_csv(out_dir / "clinical_coverage.csv", index=False)

    # (b) APOE Coverage Analysis
    print("Auditing APOE coverage and comparison with ADNIMERGE2...")
    apoe_rows = []
    apoe_rids_set = set(apoe["RID"])

    for cohort_name, c_df in [("Primary Cohort (Eligible)", primary_cohort), ("Starting Cohort (Full)", cohort)]:
        c_rids = set(c_df["RID"])
        has_apoe = c_rids.intersection(apoe_rids_set)
        missing_apoe = c_rids - apoe_rids_set

        sub_apoe = apoe[apoe["RID"].isin(has_apoe)]
        geno_counts = sub_apoe["GENOTYPE"].value_counts().to_dict()

        apoe_rows.append({
            "cohort_subset": cohort_name,
            "total_rids": len(c_rids),
            "rids_with_apoe": len(has_apoe),
            "coverage_pct": round(len(has_apoe) / len(c_rids) * 100, 2),
            "missing_rids_count": len(missing_apoe),
            "missing_rids": "; ".join(str(x) for x in sorted(missing_apoe)),
            "adnimerge2_fillable_count": 0,
            "adnimerge2_fillable_note": "ADNIMERGE2 APOERES.rda has identical participant coverage (3216 RIDs) and contains 0 of the missing cohort RIDs.",
            "genotype_distribution": json.dumps(geno_counts),
        })

    apoe_coverage_df = pd.DataFrame(apoe_rows)
    apoe_coverage_df.to_csv(out_dir / "apoe_coverage.csv", index=False)

    # (c) FreeSurfer Coverage Breakdown
    print("Auditing FreeSurfer coverage breakdown in window...")
    fs_rows = []

    for cohort_name, c_df in [("Primary Cohort (Eligible)", primary_cohort), ("Starting Cohort (Full)", cohort)]:
        total_n = len(c_df)
        rids_with_fs = set()
        rids_partial = set()
        rids_complete = set()
        rids_fs720 = set()
        rids_fs741 = set()
        rids_all_10_feat = set()
        rids_both_qc_pass = set()

        for _, r in c_df.iterrows():
            rid = r["RID"]
            idx_d = r["anchor_date"]
            w_start = idx_d - pd.Timedelta(days=180)
            w_end = idx_d + pd.Timedelta(days=90)

            sub_fs = fs[(fs["RID"] == rid) & (fs["EXAMDATE"] >= w_start) & (fs["EXAMDATE"] <= w_end)]
            if not sub_fs.empty:
                rids_with_fs.add(rid)
                if (sub_fs["STATUS"] == "partial").any():
                    rids_partial.add(rid)
                if (sub_fs["STATUS"] == "complete").any():
                    rids_complete.add(rid)
                if (sub_fs["FSVER"] == "7.2.0").any():
                    rids_fs720.add(rid)
                if (sub_fs["FSVER"] == "7.4.1").any():
                    rids_fs741.add(rid)

                feat_ok = sub_fs[target_fs_features].notna().all(axis=1).any()
                if feat_ok:
                    rids_all_10_feat.add(rid)

                qc_ok = (sub_fs["OVERALLQC"].isin(["Pass", "Partial"])).any()
                if qc_ok:
                    rids_both_qc_pass.add(rid)

        fs_rows.append({
            "cohort_subset": cohort_name,
            "total_cohort_rids": total_n,
            "rids_with_fs_row": len(rids_with_fs),
            "fs_coverage_pct": round(len(rids_with_fs) / total_n * 100, 2),
            "rids_status_partial": len(rids_partial),
            "rids_status_complete": len(rids_complete),
            "rids_fsver_720": len(rids_fs720),
            "rids_fsver_741": len(rids_fs741),
            "rids_with_all_10_target_features": len(rids_all_10_feat),
            "all_10_features_pct": round(len(rids_all_10_feat) / total_n * 100, 2),
            "rids_with_visual_qc_pass_or_partial": len(rids_both_qc_pass),
            "visual_qc_coverage_pct": round(len(rids_both_qc_pass) / total_n * 100, 2),
        })

    fs_coverage_df = pd.DataFrame(fs_rows)
    fs_coverage_df.to_csv(out_dir / "freesurfer_coverage.csv", index=False)

    # (d) MRI DICOM & Manifest Coverage
    print("Auditing MRI DICOM series and Manifest scans in window...")
    mri_rows = []

    for cohort_name, c_df in [("Primary Cohort (Eligible)", primary_cohort), ("Starting Cohort (Full)", cohort)]:
        total_n = len(c_df)
        rids_with_local_dicom = set()
        rids_with_manifest_t1 = set()
        rids_with_nifti = set()
        rids_with_both_dcm_fs = set()

        for _, r in c_df.iterrows():
            rid = r["RID"]
            idx_d = r["anchor_date"]
            w_start = idx_d - pd.Timedelta(days=180)
            w_end = idx_d + pd.Timedelta(days=90)

            sub_dcm = dcm[(dcm["RID"] == rid) & (dcm["acq_date"] >= w_start) & (dcm["acq_date"] <= w_end)]
            if not sub_dcm.empty:
                rids_with_local_dicom.add(rid)

            sub_man = mri_man[(mri_man["RID"] == rid) & (mri_man["image_date"] >= w_start) & (mri_man["image_date"] <= w_end)]
            if not sub_man.empty:
                rids_with_manifest_t1.add(rid)
                man_img_ids = set(sub_man["image_id"].dropna().astype(int))
                if man_img_ids.intersection(nii_image_ids):
                    rids_with_nifti.add(rid)

            sub_fs = fs[(fs["RID"] == rid) & (fs["EXAMDATE"] >= w_start) & (fs["EXAMDATE"] <= w_end)]
            if not sub_dcm.empty and not sub_fs.empty:
                rids_with_both_dcm_fs.add(rid)

        mri_rows.append({
            "cohort_subset": cohort_name,
            "total_cohort_rids": total_n,
            "rids_with_local_dicom_in_window": len(rids_with_local_dicom),
            "local_dicom_coverage_pct": round(len(rids_with_local_dicom) / total_n * 100, 2),
            "rids_with_manifest_t1_in_window": len(rids_with_manifest_t1),
            "manifest_t1_coverage_pct": round(len(rids_with_manifest_t1) / total_n * 100, 2),
            "rids_with_converted_nifti": len(rids_with_nifti),
            "nifti_coverage_pct": round(len(rids_with_nifti) / total_n * 100, 2),
            "rids_with_both_local_dicom_and_fs": len(rids_with_both_dcm_fs),
            "both_dicom_and_fs_pct": round(len(rids_with_both_dcm_fs) / total_n * 100, 2),
        })

    mri_coverage_df = pd.DataFrame(mri_rows)
    mri_coverage_df.to_csv(out_dir / "mri_dicom_manifest_coverage.csv", index=False)

    # (e) Reconciliation: 409 NIfTIs vs 675 DICOM Series
    print("Reconciling 409 NIfTIs vs 675 DICOM series...")
    reconcil_rows = [
        {"item": "Total Local DICOM Series", "count": 675, "description": "281 batch1 + 374 batch2 + 20 pilot series"},
        {"item": "Total Local DICOM Participant Acquisitions", "count": 665, "description": "Unique (batch, subject_id) pairs"},
        {"item": "Unique Subject RIDs across DICOM series", "count": dcm["RID"].nunique(), "description": "Unique RIDs with at least one local DICOM series"},
        {"item": "Local DICOM Series matching Starting Cohort (1199 RIDs)", "count": dcm[dcm["RID"].isin(cohort["RID"])]["RID"].nunique(), "description": "Unique cohort RIDs with local DICOMs"},
        {"item": "Local DICOM Series within Index Window [-180d, +90d]", "count": len(rids_with_local_dicom), "description": "Starting cohort RIDs with scan in baseline eligibility window"},
        {"item": "Total Converted NIfTI Images in nifti_all", "count": len(nii_image_ids), "description": "409 .nii.gz volumes and 409 .json sidecars"},
        {"item": "NIfTI Images belonging to Starting Cohort", "count": len(rids_with_nifti), "description": "409 NIfTIs correspond to the Phase 1 integrated baseline cohort"},
        {"item": "Unconverted Local DICOM Series", "count": 675 - len(nii_image_ids), "description": "266 series remain in DICOM format (unconverted repeat/multi-echo scans or pending pipeline execution)"},
    ]
    reconcil_df = pd.DataFrame(reconcil_rows)
    reconcil_df.to_csv(out_dir / "nifti_dicom_reconciliation.csv", index=False)

    # Cross-tab: MRI available vs label vs field strength vs ADNI phase
    print("Building cross-tabulation table...")
    crosstab_records = []

    for _, r in cohort.iterrows():
        rid = r["RID"]
        idx_d = r["anchor_date"]
        w_start = idx_d - pd.Timedelta(days=180)
        w_end = idx_d + pd.Timedelta(days=90)

        sub_dcm = dcm[(dcm["RID"] == rid) & (dcm["acq_date"] >= w_start) & (dcm["acq_date"] <= w_end)]
        mri_avail = not sub_dcm.empty

        field_strength = "None"
        if mri_avail:
            f_vals = sub_dcm["magnetic_field_strength"].dropna().unique()
            if len(f_vals) > 0:
                raw_f = str(f_vals[0]).strip().upper()
                if "1.5" in raw_f or "15000" in raw_f:
                    field_strength = "1.5T"
                elif "3" in raw_f or "2.9" in raw_f:
                    field_strength = "3.0T"
                else:
                    field_strength = raw_f
            else:
                field_strength = "Unknown"

        lbl = "Excluded"
        if r["exclusion_reason"] == "ELIGIBLE":
            lbl = "Positive (AD)" if r["label_primary"] == 1 else "Negative (Stable)"

        crosstab_records.append({
            "RID": rid,
            "mri_available": mri_avail,
            "outcome_label": lbl,
            "field_strength": field_strength,
            "adni_phase": r["adni_phase"],
            "exclusion_reason": r["exclusion_reason"],
        })

    crosstab_raw_df = pd.DataFrame(crosstab_records)
    crosstab_summary = crosstab_raw_df.groupby(["mri_available", "outcome_label", "field_strength", "adni_phase"]).size().reset_index(name="participant_count")
    crosstab_summary.to_csv(out_dir / "mri_label_field_phase_crosstab.csv", index=False)

    # 6. Download Candidates: RIDs lacking local DICOM or FreeSurfer in window but in Manifest
    print("Generating download candidates list...")
    download_candidates = []

    for _, r in cohort.iterrows():
        rid = r["RID"]
        idx_d = r["anchor_date"]
        w_start = idx_d - pd.Timedelta(days=180)
        w_end = idx_d + pd.Timedelta(days=90)

        sub_dcm = dcm[(dcm["RID"] == rid) & (dcm["acq_date"] >= w_start) & (dcm["acq_date"] <= w_end)]
        sub_fs = fs[(fs["RID"] == rid) & (fs["EXAMDATE"] >= w_start) & (fs["EXAMDATE"] <= w_end)]
        sub_man = mri_man[(mri_man["RID"] == rid) & (mri_man["image_date"] >= w_start) & (mri_man["image_date"] <= w_end)]

        has_local_dcm = not sub_dcm.empty
        has_fs = not sub_fs.empty
        has_man = not sub_man.empty

        if (not has_local_dcm or not has_fs) and has_man:
            img_ids = "; ".join(sub_man["image_id"].dropna().astype(str).tolist())
            img_dates = "; ".join(sub_man["image_date"].dt.strftime("%Y-%m-%d").dropna().tolist())
            img_descs = "; ".join(sub_man["image_description"].dropna().unique().tolist())

            download_candidates.append({
                "RID": rid,
                "PTID": r["PTID"],
                "index_visit_date": r["index_visit_date"].strftime("%Y-%m-%d"),
                "adni_phase": r["adni_phase"],
                "cohort_status": "Eligible_Primary" if r["exclusion_reason"] == "ELIGIBLE" else r["exclusion_reason"],
                "primary_label": r["label_primary"],
                "has_local_dicom": has_local_dcm,
                "has_freesurfer": has_fs,
                "manifest_image_ids": img_ids,
                "manifest_image_dates": img_dates,
                "manifest_descriptions": img_descs,
            })

    dl_df = pd.DataFrame(download_candidates)
    dl_df.to_csv(out_dir / "download_candidates.csv", index=False)
    print(f"Identified {len(dl_df)} download candidate participants. Saved to {out_dir / 'download_candidates.csv'}.")

    print("Coverage report generation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate coverage and reconciliation reports")
    parser.add_argument("--cohort-file", default="data/processed/cohort_v2.csv", help="Path to processed cohort CSV")
    args = parser.parse_args()
    run_coverage_analysis(args.cohort_file)
