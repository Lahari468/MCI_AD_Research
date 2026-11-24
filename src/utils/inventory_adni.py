#!/usr/bin/env python3
"""Comprehensive Inventory and Data Profiling Utility for ADNI Multimodal Datasets.
Profiles tabular clinical, FreeSurfer, genetic, and MRI metadata, as well as DICOM series headers.
Generates structured CSV files in results/inventory/.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pydicom


def compute_sha256(filepath: Path | str, chunk_size: int = 65536) -> str:
    """Compute SHA256 checksum for a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            sha.update(chunk)
    return sha.hexdigest()


def human_size(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / (1024**2):.2f} MB"
    else:
        return f"{size_bytes / (1024**3):.2f} GB"


def load_data_dictionaries() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load primary DATADIC and ADSP_PHC_DATADIC."""
    datadic_path = Path("01_raw_metadata/DATADIC_26Oct2026.csv")
    adsp_dic_path = Path("data/interim/ADSP_PHC_DATADIC.csv")

    datadic = pd.read_csv(datadic_path, low_memory=False) if datadic_path.is_file() else pd.DataFrame()
    adsp_dic = pd.read_csv(adsp_dic_path, low_memory=False) if adsp_dic_path.is_file() else pd.DataFrame()

    return datadic, adsp_dic


def find_column_description(tbl_name: str, col_name: str, datadic: pd.DataFrame, adsp_dic: pd.DataFrame) -> str:
    """Match column name to description in data dictionaries."""
    if not adsp_dic.empty and tbl_name == "ADSP_PHC_DEMODX":
        m = adsp_dic[(adsp_dic["TBLNAME"] == tbl_name) & (adsp_dic["FLDNAME"] == col_name)]
        if not m.empty:
            text = str(m.iloc[0]["TEXT"]) if pd.notna(m.iloc[0]["TEXT"]) else ""
            code = str(m.iloc[0]["CODE"]) if "CODE" in m.columns and pd.notna(m.iloc[0]["CODE"]) else ""
            return f"{text} (Codes: {code})" if code else text

    if not datadic.empty:
        # Match table and field
        m = datadic[(datadic["TBLNAME"].str.upper() == tbl_name.upper()) & (datadic["FLDNAME"].str.upper() == col_name.upper())]
        if not m.empty:
            text = str(m.iloc[0]["TEXT"]) if pd.notna(m.iloc[0]["TEXT"]) else ""
            code = str(m.iloc[0]["CODE"]) if "CODE" in m.columns and pd.notna(m.iloc[0]["CODE"]) else ""
            return f"{text} (Codes: {code})" if code else text

        # Fallback: search by field name alone
        m2 = datadic[datadic["FLDNAME"].str.upper() == col_name.upper()]
        if not m2.empty:
            text = str(m2.iloc[0]["TEXT"]) if pd.notna(m2.iloc[0]["TEXT"]) else ""
            return text

    return ""


def profile_tabular_datasets(datadic: pd.DataFrame, adsp_dic: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Profile all tabular datasets in detail."""
    tabular_files = [
        ("DXSUM", "01_raw_metadata/DXSUM_26Oct2026.csv", "clinical"),
        ("DATADIC", "01_raw_metadata/DATADIC_26Oct2026.csv", "metadata"),
        ("ADAS", "02_clinical/ADAS_26Oct2026.csv", "clinical"),
        ("CDR", "02_clinical/CDR_26Oct2026.csv", "clinical"),
        ("FAQ", "02_clinical/FAQ_26Oct2026.csv", "clinical"),
        ("MMSE", "02_clinical/MMSE_26Oct2026.csv", "clinical"),
        ("PTDEMOG", "02_clinical/PTDEMOG_26Oct2026.csv", "clinical"),
        ("UCSFFSX7", "03_freesurfer/UCSFFSX7_26Oct2026.csv", "freesurfer"),
        ("APOERES", "05_genetics/APOERES_26Oct2026.csv", "genetics"),
        ("ADSP_PHC_DEMODX", "data/interim/ADSP_PHC_DEMODX.csv", "clinical"),
        ("ADSP_PHC_DATADIC", "data/interim/ADSP_PHC_DATADIC.csv", "metadata"),
        ("T1w_Manifest", "04_mri_metadata/T1w_Structural_MRI_Cohort_Manifest_26Oct2026.csv", "mri_metadata"),
        ("T1w_URL_List", "04_mri_metadata/T1w_Structural_MRI_Cohort_URL_List.csv", "mri_metadata"),
        ("idaSearch", "04_mri_metadata/idaSearch_10_27_2026 (2).csv", "mri_metadata"),
        ("IDA_batch_match_summary", "04_mri_metadata/IDA_batch_match_summary.csv", "mri_metadata"),
        ("IDA_batch_unmatched_date_audit", "04_mri_metadata/IDA_batch_unmatched_date_audit.csv", "mri_metadata"),
        ("IDA_batch_unmatched_records", "04_mri_metadata/IDA_batch_unmatched_records.csv", "mri_metadata"),
        ("IDA_batch_verified_matches", "04_mri_metadata/IDA_batch_verified_matches.csv", "mri_metadata"),
        ("IDA_image_id_test_batch_20", "04_mri_metadata/IDA_image_id_test_batch_20.csv", "mri_metadata"),
        ("MRI_batch1_image_ids", "04_mri_metadata/MRI_batch1_image_ids.csv", "mri_metadata"),
    ]

    summary_rows = []
    column_rows = []

    id_patterns = [r"^RID$", r"^PTID$", r"^SUBJID$", r"^SUBJECT.*ID$", r"^ID$", r"^IMAGE.*ID$"]
    visit_patterns = [r"^VISCODE.*", r"^PHC_VISIT$", r"^IMAGE_VISIT$", r"^VISDATE$"]
    date_patterns = [r".*DATE.*", r"^USERDATE.*", r"^APTESTDT$", r"^MMYEAR$", r"^EXAMDATE$"]

    for table_name, rel_path, category in tabular_files:
        p = Path(rel_path)
        if not p.is_file():
            print(f"Skipping missing file: {rel_path}")
            continue

        print(f"Profiling tabular file: {table_name} ({rel_path})...")
        file_size = p.stat().st_size
        sha = compute_sha256(p)

        df = pd.read_csv(p, low_memory=False)
        row_count, col_count = df.shape

        candidate_ids = [c for c in df.columns if any(re.match(pat, c, re.IGNORECASE) for pat in id_patterns)]
        candidate_visits = [c for c in df.columns if any(re.match(pat, c, re.IGNORECASE) for pat in visit_patterns)]
        candidate_dates = [c for c in df.columns if any(re.match(pat, c, re.IGNORECASE) for pat in date_patterns)]

        # Duplicate keys test
        key_dups = {}
        for id_col in ["RID", "PTID", "Subject ID", "subject_id"]:
            if id_col in df.columns:
                for v_col in ["VISCODE2", "VISCODE", "EXAMDATE", "image_date", "Study Date"]:
                    if v_col in df.columns:
                        dup_c = df.duplicated(subset=[id_col, v_col]).sum()
                        key_dups[f"({id_col}, {v_col})"] = int(dup_c)

        total_cells = row_count * col_count
        total_nulls = int(df.isna().sum().sum())
        missing_rate = round((total_nulls / total_cells * 100), 2) if total_cells > 0 else 0.0

        summary_rows.append({
            "table_name": table_name,
            "category": category,
            "file_path": str(p),
            "size_bytes": file_size,
            "size_human": human_size(file_size),
            "sha256": sha,
            "row_count": row_count,
            "column_count": col_count,
            "candidate_id_cols": "; ".join(candidate_ids),
            "candidate_visit_cols": "; ".join(candidate_visits),
            "candidate_date_cols": "; ".join(candidate_dates),
            "duplicate_key_counts": json.dumps(key_dups),
            "total_null_cells": total_nulls,
            "overall_missing_pct": missing_rate,
            "data_dictionary_source": "ADSP_PHC_DATADIC" if "ADSP_PHC" in table_name else ("DATADIC" if table_name in ["ADAS", "CDR", "FAQ", "MMSE", "PTDEMOG", "DXSUM", "UCSFFSX7", "APOERES"] else "None"),
        })

        # Columns profile
        for col in df.columns:
            s = df[col]
            n_null = int(s.isna().sum())
            n_valid = row_count - n_null
            null_pct = round(n_null / row_count * 100, 2) if row_count > 0 else 0.0
            
            # Sample values (up to 3 distinct non-null)
            samples = s.dropna().unique()[:3]
            sample_str = ", ".join(str(x) for x in samples)

            desc = find_column_description(table_name, col, datadic, adsp_dic)

            column_rows.append({
                "table_name": table_name,
                "column_name": col,
                "dtype": str(s.dtype),
                "total_rows": row_count,
                "non_null_count": n_valid,
                "null_count": n_null,
                "null_rate_percent": null_pct,
                "is_id_candidate": any(re.match(pat, col, re.IGNORECASE) for pat in id_patterns),
                "is_visit_candidate": any(re.match(pat, col, re.IGNORECASE) for pat in visit_patterns),
                "is_date_candidate": any(re.match(pat, col, re.IGNORECASE) for pat in date_patterns),
                "sample_values": sample_str,
                "dictionary_description": desc,
            })

    summary_df = pd.DataFrame(summary_rows)
    columns_df = pd.DataFrame(column_rows)

    summary_df.to_csv(out_dir / "tabular_summary.csv", index=False)
    columns_df.to_csv(out_dir / "tabular_columns_detailed.csv", index=False)

    return summary_df, columns_df


def profile_freesurfer(out_dir: Path) -> pd.DataFrame:
    """Detailed audit of UCSFFSX7_26Oct2026.csv."""
    p = Path("03_freesurfer/UCSFFSX7_26Oct2026.csv")
    df = pd.read_csv(p, low_memory=False)

    qc_cols = ["OVERALLQC", "TEMPQC", "FRONTQC", "PARQC", "INSULAQC", "OCCQC", "BGQC", "CWMQC", "VENTQC", "HIPPOQC"]
    meta_cols = ["STATUS", "FSVER"]

    audit_rows = []
    for col in meta_cols + qc_cols:
        val_counts = df[col].value_counts(dropna=False).to_dict()
        n_missing = int(df[col].isna().sum())
        pct_missing = round(n_missing / len(df) * 100, 2)
        audit_rows.append({
            "column_name": col,
            "type": "Metadata" if col in meta_cols else "Regional QC",
            "non_null_count": int(df[col].notna().sum()),
            "missing_count": n_missing,
            "missing_rate_pct": pct_missing,
            "distribution": json.dumps({str(k): int(v) for k, v in val_counts.items()}),
        })

    # Cross-tabulation of STATUS by FSVER
    crosstab_sf = pd.crosstab(df["STATUS"], df["FSVER"])
    print("FreeSurfer STATUS by FSVER:\n", crosstab_sf)

    # QC completeness by STATUS
    for status_val in df["STATUS"].unique():
        sub = df[df["STATUS"] == status_val]
        qc_filled = sub["OVERALLQC"].notna().sum()
        audit_rows.append({
            "column_name": f"QC_Completeness_when_STATUS={status_val}",
            "type": "Stratified QC Completeness",
            "non_null_count": int(qc_filled),
            "missing_count": int(len(sub) - qc_filled),
            "missing_rate_pct": round((len(sub) - qc_filled) / len(sub) * 100, 2),
            "distribution": json.dumps({"total_rows": len(sub), "qc_present": int(qc_filled)}),
        })

    res_df = pd.DataFrame(audit_rows)
    res_df.to_csv(out_dir / "freesurfer_ucsffsx7_qc_audit.csv", index=False)
    return res_df


def profile_clinical_assessments(out_dir: Path) -> pd.DataFrame:
    """Deep dive audit into MMSE, ADAS, CDR, FAQ, PTDEMOG, APOE, and ADSP_PHC_DEMODX."""
    audit_records = []

    # 1. MMSE
    mmse = pd.read_csv("02_clinical/MMSE_26Oct2026.csv", low_memory=False)
    audit_records.append({
        "instrument": "MMSE",
        "primary_metric": "MMSCORE",
        "total_records": len(mmse),
        "unique_participants": int(mmse["RID"].nunique()),
        "metric_non_null": int(mmse["MMSCORE"].notna().sum()),
        "metric_missing_pct": round(mmse["MMSCORE"].isna().mean() * 100, 2),
        "metric_min": float(mmse["MMSCORE"].min()),
        "metric_median": float(mmse["MMSCORE"].median()),
        "metric_max": float(mmse["MMSCORE"].max()),
        "subscales_present": "Orientation (MMYEAR, MMDAY..), Registration, Attention (WORLDSCORE/CALCSCORE), Recall, Language, Visuospatial",
        "key_notes": "Scored 0-30. Scores <24 indicate cognitive impairment. Both raw subitems and composite MMSCORE available.",
    })

    # 2. ADAS
    adas = pd.read_csv("02_clinical/ADAS_26Oct2026.csv", low_memory=False)
    score_col = "TOTSCORE" if "TOTSCORE" in adas.columns else ("TOTAL13" if "TOTAL13" in adas.columns else adas.columns[10])
    tot13_col = "TOTAL13" if "TOTAL13" in adas.columns else None
    audit_records.append({
        "instrument": "ADAS-Cog",
        "primary_metric": f"{score_col} (Total Score) & {tot13_col} (ADAS-13)",
        "total_records": len(adas),
        "unique_participants": int(adas["RID"].nunique()),
        "metric_non_null": int(adas[score_col].notna().sum()),
        "metric_missing_pct": round(adas[score_col].isna().mean() * 100, 2),
        "metric_min": float(adas[score_col].min()),
        "metric_median": float(adas[score_col].median()),
        "metric_max": float(adas[score_col].max()),
        "subscales_present": "Q1SCORE to Q13SCORE (Word recall, Commands, Construction, Naming, Ideational, Orientation, Word Recog, Instructions, Spoken Lang, Word Finding, Comprehension, Delayed Recall, Number Cancellation)",
        "key_notes": "TOTSCORE represents overall total score (typically ADAS-11 basis), TOTAL13 represents 13-item extended score (range 0-85). Major cognitive endpoint.",
    })

    # 3. CDR
    cdr = pd.read_csv("02_clinical/CDR_26Oct2026.csv", low_memory=False)
    sb_metric = "CDRSB" if "CDRSB" in cdr.columns else "CDGLOBAL"
    audit_records.append({
        "instrument": "CDR",
        "primary_metric": f"CDGLOBAL (Global CDR) & {sb_metric} (Sum of Boxes)",
        "total_records": len(cdr),
        "unique_participants": int(cdr["RID"].nunique()),
        "metric_non_null": int(cdr["CDGLOBAL"].notna().sum()),
        "metric_missing_pct": round(cdr["CDGLOBAL"].isna().mean() * 100, 2),
        "metric_min": float(cdr["CDGLOBAL"].min()),
        "metric_median": float(cdr["CDGLOBAL"].median()),
        "metric_max": float(cdr["CDGLOBAL"].max()),
        "subscales_present": "CDMEMORY, CDORIENT, CDJUDGE, CDCOMMUN, CDHOME, CDCARE",
        "key_notes": "CDGLOBAL: 0=Normal, 0.5=MCI/Very Mild, 1=Mild, 2=Moderate, 3=Severe. CDRSB provides fine-grained 0-18 scale directly in table.",
    })

    # 4. FAQ
    faq = pd.read_csv("02_clinical/FAQ_26Oct2026.csv", low_memory=False)
    audit_records.append({
        "instrument": "FAQ",
        "primary_metric": "FAQTOTAL",
        "total_records": len(faq),
        "unique_participants": int(faq["RID"].nunique()),
        "metric_non_null": int(faq["FAQTOTAL"].notna().sum()),
        "metric_missing_pct": round(faq["FAQTOTAL"].isna().mean() * 100, 2),
        "metric_min": float(faq["FAQTOTAL"].min()),
        "metric_median": float(faq["FAQTOTAL"].median()),
        "metric_max": float(faq["FAQTOTAL"].max()),
        "subscales_present": "Financial, Forms/Taxes, Shopping, Games/Hobbies, Stove/Beverages, Meal Prep, Current Events, TV/Books, Remembering, Traveling",
        "key_notes": "Range 0-30. Scores >=9 indicate functional impairment / loss of independence characteristic of dementia.",
    })

    # 5. PTDEMOG
    demog = pd.read_csv("02_clinical/PTDEMOG_26Oct2026.csv", low_memory=False)
    audit_records.append({
        "instrument": "PTDEMOG",
        "primary_metric": "PTGENDER, PTEDUCAT, PTETHCAT, PTRACCAT, PTMARRY",
        "total_records": len(demog),
        "unique_participants": int(demog["RID"].nunique()),
        "metric_non_null": int(demog["PTEDUCAT"].notna().sum()),
        "metric_missing_pct": round(demog["PTEDUCAT"].isna().mean() * 100, 2),
        "metric_min": float(demog["PTEDUCAT"].min()),
        "metric_median": float(demog["PTEDUCAT"].median()),
        "metric_max": float(demog["PTEDUCAT"].max()),
        "subscales_present": "Gender (1=Male, 2=Female), Education (years: 0-20), Race, Ethnicity, Marital Status",
        "key_notes": "Baseline demographic information. Education years is a major cognitive reserve proxy.",
    })

    # 6. APOE
    apoe = pd.read_csv("05_genetics/APOERES_26Oct2026.csv", low_memory=False)
    val_counts = apoe["GENOTYPE"].value_counts().to_dict()
    audit_records.append({
        "instrument": "APOE",
        "primary_metric": "GENOTYPE (e.g. 3/3, 3/4, 4/4)",
        "total_records": len(apoe),
        "unique_participants": int(apoe["RID"].nunique()),
        "metric_non_null": int(apoe["GENOTYPE"].notna().sum()),
        "metric_missing_pct": round(apoe["GENOTYPE"].isna().mean() * 100, 2),
        "metric_min": np.nan,
        "metric_median": np.nan,
        "metric_max": np.nan,
        "subscales_present": f"Genotype breakdown: {json.dumps(val_counts)}",
        "key_notes": "Carriers of APOE epsilon 4 (3/4, 4/4, 2/4) have substantially higher risk of progression to AD.",
    })

    # 7. ADSP_PHC_DEMODX
    adsp = pd.read_csv("data/interim/ADSP_PHC_DEMODX.csv", low_memory=False)
    dx_counts = adsp["PHC_Diagnosis"].value_counts().to_dict()
    audit_records.append({
        "instrument": "ADSP_PHC_DEMODX",
        "primary_metric": "PHC_Diagnosis (1=CN, 2=MCI, 3=AD, 4=non-AD, 5=Other)",
        "total_records": len(adsp),
        "unique_participants": int(adsp["RID"].nunique()),
        "metric_non_null": int(adsp["PHC_Diagnosis"].notna().sum()),
        "metric_missing_pct": round(adsp["PHC_Diagnosis"].isna().mean() * 100, 2),
        "metric_min": float(adsp["PHC_Diagnosis"].min()),
        "metric_median": float(adsp["PHC_Diagnosis"].median()),
        "metric_max": float(adsp["PHC_Diagnosis"].max()),
        "subscales_present": f"Diagnostic classes: {json.dumps(dx_counts)}",
        "key_notes": "Harmonized diagnostic outcome standard. PHC_Diagnosis=2 at baseline defines study entry; transition to 3 defines primary AD endpoint.",
    })

    res_df = pd.DataFrame(audit_records)
    res_df.to_csv(out_dir / "clinical_assessments_detailed.csv", index=False)
    return res_df


def profile_dicom_series(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse headers across all DICOM series across batch1, batch2, and pilot."""
    dicom_roots = [
        ("batch1_combined", Path("04_mri_metadata/batch1_combined/ADNI")),
        ("batch2_combined", Path("04_mri_metadata/batch2_combined/ADNI")),
        ("pilot_download", Path("04_mri_metadata/pilot_download/ADNI")),
    ]

    series_records = []

    for batch_name, root_path in dicom_roots:
        if not root_path.exists():
            continue
        print(f"Scanning DICOM series in {batch_name} ({root_path})...")

        for subject_dir in sorted(root_path.iterdir()):
            if not subject_dir.is_dir() or subject_dir.name.startswith("."):
                continue

            subj_id_from_dir = subject_dir.name

            for root, dirs, files in os.walk(subject_dir):
                dcm_files = [f for f in files if f.endswith(".dcm") or (not "." in f and not f.startswith("."))]
                if not dcm_files:
                    continue

                # Parse header of first file
                first_file = os.path.join(root, dcm_files[0])
                try:
                    ds = pydicom.dcmread(first_file, stop_before_pixels=True)
                except Exception as e:
                    print(f"Error reading DICOM {first_file}: {e}")
                    continue

                # Extract key fields
                patient_id = str(ds.get("PatientID", subj_id_from_dir))
                series_desc = str(ds.get("SeriesDescription", "Unknown"))
                series_num = str(ds.get("SeriesNumber", ""))
                study_date = str(ds.get("StudyDate", ds.get("AcquisitionDate", "")))
                mfg = str(ds.get("Manufacturer", "Unknown"))
                model = str(ds.get("ManufacturerModelName", "Unknown"))
                field = ds.get("MagneticFieldStrength", None)
                field_str = f"{float(field):.1f}T" if field is not None else "Unknown"

                tr = float(ds.get("RepetitionTime", 0)) if ds.get("RepetitionTime") is not None else np.nan
                te = float(ds.get("EchoTime", 0)) if ds.get("EchoTime") is not None else np.nan
                ti = float(ds.get("InversionTime", 0)) if ds.get("InversionTime") is not None else np.nan
                flip = float(ds.get("FlipAngle", 0)) if ds.get("FlipAngle") is not None else np.nan

                # Voxel size
                pix_spacing = ds.get("PixelSpacing", [np.nan, np.nan])
                dx = float(pix_spacing[0]) if len(pix_spacing) > 0 else np.nan
                dy = float(pix_spacing[1]) if len(pix_spacing) > 1 else np.nan
                dz = float(ds.get("SliceThickness", np.nan)) if ds.get("SliceThickness") is not None else np.nan
                voxel_str = f"{dx:.2f} x {dy:.2f} x {dz:.2f} mm" if not np.isnan(dx) else "Unknown"

                rows = int(ds.get("Rows", 0)) if ds.get("Rows") is not None else 0
                cols = int(ds.get("Columns", 0)) if ds.get("Columns") is not None else 0

                sample_sha = compute_sha256(first_file)

                series_records.append({
                    "batch": batch_name,
                    "subject_id": patient_id,
                    "subject_dir": subj_id_from_dir,
                    "series_description": series_desc,
                    "series_number": series_num,
                    "acquisition_date": study_date,
                    "manufacturer": mfg,
                    "model_name": model,
                    "magnetic_field_strength": field_str,
                    "repetition_time_tr": tr,
                    "echo_time_te": te,
                    "inversion_time_ti": ti,
                    "flip_angle": flip,
                    "pixel_spacing_x": dx,
                    "pixel_spacing_y": dy,
                    "slice_thickness_z": dz,
                    "voxel_size": voxel_str,
                    "matrix_rows": rows,
                    "matrix_columns": cols,
                    "slice_count": len(dcm_files),
                    "series_path": root,
                    "sample_file_sha256": sample_sha,
                })

    df_series = pd.DataFrame(series_records)
    df_series.to_csv(out_dir / "dicom_series_headers.csv", index=False)

    # Subject summary
    subject_summary = []
    for (batch, subj), g in df_series.groupby(["batch", "subject_id"]):
        subject_summary.append({
            "batch": batch,
            "subject_id": subj,
            "series_count": len(g),
            "series_descriptions": "; ".join(g["series_description"].unique()),
            "manufacturers": "; ".join(g["manufacturer"].unique()),
            "field_strengths": "; ".join(g["magnetic_field_strength"].unique()),
            "total_slices": int(g["slice_count"].sum()),
            "earliest_date": str(g["acquisition_date"].min()),
            "latest_date": str(g["acquisition_date"].max()),
        })

    df_subjects = pd.DataFrame(subject_summary)
    df_subjects.to_csv(out_dir / "dicom_subject_summary.csv", index=False)

    return df_series, df_subjects


def build_complete_file_manifest(out_dir: Path) -> pd.DataFrame:
    """Build complete file manifest of all downloaded and raw files."""
    manifest_rows = []

    # 1. Tabular files
    search_paths = [
        ("01_raw_metadata", "metadata"),
        ("02_clinical", "clinical"),
        ("03_freesurfer", "freesurfer"),
        ("04_mri_metadata", "mri_metadata"),
        ("05_genetics", "genetics"),
        ("data/interim", "interim"),
    ]

    for dir_path, cat in search_paths:
        p = Path(dir_path)
        if not p.exists():
            continue
        for f in p.iterdir():
            if f.is_file() and not f.name.startswith("."):
                ext = f.suffix.upper().replace(".", "")
                ftype = "CSV" if ext == "CSV" else ("TAR_GZ" if f.name.endswith(".tar.gz") else ext)
                size_b = f.stat().st_size
                sha = compute_sha256(f)

                row_cnt = None
                col_cnt = None
                if ftype == "CSV":
                    try:
                        df_tmp = pd.read_csv(f, low_memory=False)
                        row_cnt, col_cnt = df_tmp.shape
                    except Exception:
                        pass

                manifest_rows.append({
                    "file_path": str(f),
                    "file_name": f.name,
                    "category": cat,
                    "file_type": ftype,
                    "size_bytes": size_b,
                    "size_human": human_size(size_b),
                    "sha256": sha,
                    "row_count": row_cnt,
                    "column_count": col_cnt,
                })

    # 2. NIfTI conversions
    nifti_dir = Path("04_mri_metadata/nifti_all")
    if nifti_dir.exists():
        nii_files = list(nifti_dir.glob("*.nii.gz"))
        json_files = list(nifti_dir.glob("*.json"))
        # Summary row for NIfTI
        total_nii_size = sum(f.stat().st_size for f in nii_files)
        total_json_size = sum(f.stat().st_size for f in json_files)
        manifest_rows.append({
            "file_path": str(nifti_dir),
            "file_name": f"nifti_all ({len(nii_files)} NIfTI volumes + {len(json_files)} JSON sidecars)",
            "category": "mri_nifti",
            "file_type": "NIFTI_COLLECTION",
            "size_bytes": total_nii_size + total_json_size,
            "size_human": human_size(total_nii_size + total_json_size),
            "sha256": "N/A (Collection of 410 image-json pairs)",
            "row_count": len(nii_files),
            "column_count": 2,
        })

    # 3. DICOM Collections
    for batch_name, p in [
        ("batch1_combined", Path("04_mri_metadata/batch1_combined/ADNI")),
        ("batch2_combined", Path("04_mri_metadata/batch2_combined/ADNI")),
        ("pilot_download", Path("04_mri_metadata/pilot_download/ADNI")),
    ]:
        if p.exists():
            dcm_count = 0
            dcm_size = 0
            for root, _, files in os.walk(p):
                for f in files:
                    if f.endswith(".dcm") or (not "." in f and not f.startswith(".")):
                        dcm_count += 1
                        dcm_size += os.path.getsize(os.path.join(root, f))
            manifest_rows.append({
                "file_path": str(p),
                "file_name": f"{batch_name}/ADNI ({dcm_count} DICOM slices)",
                "category": "mri_dicom",
                "file_type": "DICOM_DATASET",
                "size_bytes": dcm_size,
                "size_human": human_size(dcm_size),
                "sha256": f"N/A ({dcm_count} files)",
                "row_count": dcm_count,
                "column_count": None,
            })

    df_manifest = pd.DataFrame(manifest_rows)
    df_manifest.to_csv(out_dir / "file_manifest.csv", index=False)
    return df_manifest


def main():
    out_dir = Path("results/inventory")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Starting ADNI multimodal dataset inventory into {out_dir}...")

    # Load dictionaries
    datadic, adsp_dic = load_data_dictionaries()
    print(f"Loaded DATADIC ({len(datadic)} rows) and ADSP_PHC_DATADIC ({len(adsp_dic)} rows).")

    # 1. Profile tabular files
    summary_df, columns_df = profile_tabular_datasets(datadic, adsp_dic, out_dir)
    print(f"Profiled {len(summary_df)} tabular datasets and {len(columns_df)} individual columns.")

    # 2. Profile FreeSurfer QC and versions
    fs_df = profile_freesurfer(out_dir)
    print(f"Profiled FreeSurfer QC/status metrics ({len(fs_df)} audit metrics).")

    # 3. Profile clinical assessments
    clin_df = profile_clinical_assessments(out_dir)
    print(f"Profiled clinical instruments ({len(clin_df)} instruments).")

    # 4. Profile DICOM series headers
    dicom_series_df, dicom_subj_df = profile_dicom_series(out_dir)
    print(f"Profiled {len(dicom_series_df)} DICOM series across {len(dicom_subj_df)} subject acquisitions.")

    # 5. Build complete file manifest
    manifest_df = build_complete_file_manifest(out_dir)
    print(f"Generated complete file manifest ({len(manifest_df)} file/collection entries).")

    print("ADNI Inventory extraction complete!")


if __name__ == "__main__":
    main()
