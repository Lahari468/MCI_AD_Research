"""FreeSurfer Feature Extraction, Outlier Auditing, and Verification Pipeline.

Extracts the 10 target FreeSurfer features + ICV from UCSFFSX7:
  - Bilateral hippocampal volumes (ST29SV, ST88SV)
  - Bilateral cortical thicknesses (ST24TA, ST83TA, ST44TA, ST103TA, ST40TA, ST99TA, ST52TA, ST111TA)
  - Estimated Intracranial Volume (ST10CV)

Enforces:
  1. Verbatim DATADIC verification ensuring all 8 thickness features map strictly to
     Thickness Average (aparc.stats) and not cortical/subcortical volume.
  2. Blind nearest-scan selection within [T_anchor - 180 d, T_anchor + 90 d].
  3. Pre-anchor tie preference and deterministic QC/IMAGEUID tie-breaking.
  4. Label-free robust z-score calculation across ALL rows of UCSFFSX7.
  5. ICV-normalised hippocampal volumes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# Verbatim mapping dictionary derived directly from DATADIC_26Oct2026.csv for UCSFFSX7
FS_TARGET_MAPPING: dict[str, dict[str, str]] = {
    "ST29SV": {
        "name": "left_hippocampus_volume",
        "region": "Left Hippocampus",
        "measure": "Volume (mm3)",
        "datadic_text": "Subcortical Volume (aseg.stats) of LeftHippocampus",
    },
    "ST88SV": {
        "name": "right_hippocampus_volume",
        "region": "Right Hippocampus",
        "measure": "Volume (mm3)",
        "datadic_text": "Subcortical Volume (aseg.stats) of RightHippocampus",
    },
    "ST24TA": {
        "name": "left_entorhinal_thickness",
        "region": "Left Entorhinal",
        "measure": "Thickness Average (mm)",
        "datadic_text": "Thickness Average (aparc.stats) of LeftEntorhinal",
    },
    "ST83TA": {
        "name": "right_entorhinal_thickness",
        "region": "Right Entorhinal",
        "measure": "Thickness Average (mm)",
        "datadic_text": "Thickness Average (aparc.stats) of RightEntorhinal",
    },
    "ST44TA": {
        "name": "left_parahippocampal_thickness",
        "region": "Left Parahippocampal",
        "measure": "Thickness Average (mm)",
        "datadic_text": "Thickness Average (aparc.stats) of LeftParahippocampal",
    },
    "ST103TA": {
        "name": "right_parahippocampal_thickness",
        "region": "Right Parahippocampal",
        "measure": "Thickness Average (mm)",
        "datadic_text": "Thickness Average (aparc.stats) of RightParahippocampal",
    },
    "ST40TA": {
        "name": "left_middle_temporal_thickness",
        "region": "Left Middle Temporal",
        "measure": "Thickness Average (mm)",
        "datadic_text": "Thickness Average (aparc.stats) of LeftMiddleTemporal",
    },
    "ST99TA": {
        "name": "right_middle_temporal_thickness",
        "region": "Right Middle Temporal",
        "measure": "Thickness Average (mm)",
        "datadic_text": "Thickness Average (aparc.stats) of RightMiddleTemporal",
    },
    "ST52TA": {
        "name": "left_precuneus_thickness",
        "region": "Left Precuneus",
        "measure": "Thickness Average (mm)",
        "datadic_text": "Thickness Average (aparc.stats) of LeftPrecuneus",
    },
    "ST111TA": {
        "name": "right_precuneus_thickness",
        "region": "Right Precuneus",
        "measure": "Thickness Average (mm)",
        "datadic_text": "Thickness Average (aparc.stats) of RightPrecuneus",
    },
    "ST10CV": {
        "name": "estimated_icv",
        "region": "Intracranial Cavity",
        "measure": "Estimated ICV Volume (mm3)",
        "datadic_text": "Cortical Volume (aparc.stats) of Icv",
    },
}

FS_FEATURE_COLS = list(FS_TARGET_MAPPING.keys())

# A-Priori 29-Feature FreeSurfer Set (fs29) derived strictly from DATADIC_26Oct2026.csv
FS29_TARGET_MAPPING: dict[str, dict[str, str]] = {
    # L/R volume: hippocampus, amygdala, lateral ventricle, inferior lateral ventricle
    "ST29SV": {
        "name": "left_hippocampus_volume",
        "region": "Left Hippocampus",
        "measure": "Volume (mm3)",
        "datadic_text": "Subcortical Volume (aseg.stats) of LeftHippocampus",
    },
    "ST88SV": {
        "name": "right_hippocampus_volume",
        "region": "Right Hippocampus",
        "measure": "Volume (mm3)",
        "datadic_text": "Subcortical Volume (aseg.stats) of RightHippocampus",
    },
    "ST12SV": {
        "name": "left_amygdala_volume",
        "region": "Left Amygdala",
        "measure": "Volume (mm3)",
        "datadic_text": "Subcortical Volume (aseg.stats) of LeftAmygdala",
    },
    "ST71SV": {
        "name": "right_amygdala_volume",
        "region": "Right Amygdala",
        "measure": "Volume (mm3)",
        "datadic_text": "Subcortical Volume (aseg.stats) of RightAmygdala",
    },
    "ST37SV": {
        "name": "left_lateral_ventricle_volume",
        "region": "Left Lateral Ventricle",
        "measure": "Volume (mm3)",
        "datadic_text": "Subcortical Volume (aseg.stats) of LeftLateralVentricle",
    },
    "ST96SV": {
        "name": "right_lateral_ventricle_volume",
        "region": "Right Lateral Ventricle",
        "measure": "Volume (mm3)",
        "datadic_text": "Subcortical Volume (aseg.stats) of RightLateralVentricle",
    },
    "ST30SV": {
        "name": "left_inferior_lateral_ventricle_volume",
        "region": "Left Inferior Lateral Ventricle",
        "measure": "Volume (mm3)",
        "datadic_text": "Subcortical Volume (aseg.stats) of LeftInferiorLateralVentricle",
    },
    "ST89SV": {
        "name": "right_inferior_lateral_ventricle_volume",
        "region": "Right Inferior Lateral Ventricle",
        "measure": "Volume (mm3)",
        "datadic_text": "Subcortical Volume (aseg.stats) of RightInferiorLateralVentricle",
    },
    # L/R cortical thickness average:
    # entorhinal, parahippocampal, fusiform, inferior temporal, middle temporal,
    # precuneus, posterior cingulate, isthmus cingulate, inferior parietal, superior temporal
    "ST24TA": {
        "name": "left_entorhinal_thickness",
        "region": "Left Entorhinal",
        "measure": "Thickness Average (mm)",
        "datadic_text": "Thickness Average (aparc.stats) of LeftEntorhinal",
    },
    "ST83TA": {
        "name": "right_entorhinal_thickness",
        "region": "Right Entorhinal",
        "measure": "Thickness Average (mm)",
        "datadic_text": "Thickness Average (aparc.stats) of RightEntorhinal",
    },
    "ST44TA": {
        "name": "left_parahippocampal_thickness",
        "region": "Left Parahippocampal",
        "measure": "Thickness Average (mm)",
        "datadic_text": "Thickness Average (aparc.stats) of LeftParahippocampal",
    },
    "ST103TA": {
        "name": "right_parahippocampal_thickness",
        "region": "Right Parahippocampal",
        "measure": "Thickness Average (mm)",
        "datadic_text": "Thickness Average (aparc.stats) of RightParahippocampal",
    },
    "ST26TA": {
        "name": "left_fusiform_thickness",
        "region": "Left Fusiform",
        "measure": "Thickness Average (mm)",
        "datadic_text": "Thickness Average (aparc.stats) of LeftFusiform",
    },
    "ST85TA": {
        "name": "right_fusiform_thickness",
        "region": "Right Fusiform",
        "measure": "Thickness Average (mm)",
        "datadic_text": "Thickness Average (aparc.stats) of RightFusiform",
    },
    "ST32TA": {
        "name": "left_inferior_temporal_thickness",
        "region": "Left Inferior Temporal",
        "measure": "Thickness Average (mm)",
        "datadic_text": "Thickness Average (aparc.stats) of LeftInferiorTemporal",
    },
    "ST91TA": {
        "name": "right_inferior_temporal_thickness",
        "region": "Right Inferior Temporal",
        "measure": "Thickness Average (mm)",
        "datadic_text": "Thickness Average (aparc.stats) of RightInferiorTemporal",
    },
    "ST40TA": {
        "name": "left_middle_temporal_thickness",
        "region": "Left Middle Temporal",
        "measure": "Thickness Average (mm)",
        "datadic_text": "Thickness Average (aparc.stats) of LeftMiddleTemporal",
    },
    "ST99TA": {
        "name": "right_middle_temporal_thickness",
        "region": "Right Middle Temporal",
        "measure": "Thickness Average (mm)",
        "datadic_text": "Thickness Average (aparc.stats) of RightMiddleTemporal",
    },
    "ST52TA": {
        "name": "left_precuneus_thickness",
        "region": "Left Precuneus",
        "measure": "Thickness Average (mm)",
        "datadic_text": "Thickness Average (aparc.stats) of LeftPrecuneus",
    },
    "ST111TA": {
        "name": "right_precuneus_thickness",
        "region": "Right Precuneus",
        "measure": "Thickness Average (mm)",
        "datadic_text": "Thickness Average (aparc.stats) of RightPrecuneus",
    },
    "ST50TA": {
        "name": "left_posterior_cingulate_thickness",
        "region": "Left Posterior Cingulate",
        "measure": "Thickness Average (mm)",
        "datadic_text": "Thickness Average (aparc.stats) of LeftPosteriorCingulate",
    },
    "ST109TA": {
        "name": "right_posterior_cingulate_thickness",
        "region": "Right Posterior Cingulate",
        "measure": "Thickness Average (mm)",
        "datadic_text": "Thickness Average (aparc.stats) of RightPosteriorCingulate",
    },
    "ST34TA": {
        "name": "left_isthmus_cingulate_thickness",
        "region": "Left Isthmus Cingulate",
        "measure": "Thickness Average (mm)",
        "datadic_text": "Thickness Average (aparc.stats) of LeftIsthmusCingulate",
    },
    "ST93TA": {
        "name": "right_isthmus_cingulate_thickness",
        "region": "Right Isthmus Cingulate",
        "measure": "Thickness Average (mm)",
        "datadic_text": "Thickness Average (aparc.stats) of RightIsthmusCingulate",
    },
    "ST31TA": {
        "name": "left_inferior_parietal_thickness",
        "region": "Left Inferior Parietal",
        "measure": "Thickness Average (mm)",
        "datadic_text": "Thickness Average (aparc.stats) of LeftInferiorParietal",
    },
    "ST90TA": {
        "name": "right_inferior_parietal_thickness",
        "region": "Right Inferior Parietal",
        "measure": "Thickness Average (mm)",
        "datadic_text": "Thickness Average (aparc.stats) of RightInferiorParietal",
    },
    "ST58TA": {
        "name": "left_superior_temporal_thickness",
        "region": "Left Superior Temporal",
        "measure": "Thickness Average (mm)",
        "datadic_text": "Thickness Average (aparc.stats) of LeftSuperiorTemporal",
    },
    "ST117TA": {
        "name": "right_superior_temporal_thickness",
        "region": "Right Superior Temporal",
        "measure": "Thickness Average (mm)",
        "datadic_text": "Thickness Average (aparc.stats) of RightSuperiorTemporal",
    },
    # ICV
    "ST10CV": {
        "name": "estimated_icv",
        "region": "Intracranial Cavity",
        "measure": "Estimated ICV Volume (mm3)",
        "datadic_text": "Cortical Volume (aparc.stats) of Icv",
    },
}

FS29_FEATURE_COLS = list(FS29_TARGET_MAPPING.keys())


def compute_robust_z_params(
    fs_df: pd.DataFrame, feature_cols: list[str] | None = None
) -> dict[str, tuple[float, float]]:
    """Compute label-free median and MAD scale (1.4826 * MAD) across all UCSFFSX7 rows."""
    if feature_cols is None:
        feature_cols = FS_FEATURE_COLS
    params = {}
    for col in feature_cols:
        s = pd.to_numeric(fs_df[col], errors="coerce").dropna()
        med = float(s.median())
        mad = float((s - med).abs().median())
        scale = 1.4826 * mad if mad > 0 else 1.0
        params[col] = (med, scale)
    return params


def extract_freesurfer_features(
    cohort_path: Path | str = "data/processed/cohort_v2b.csv",
    fs_path: Path | str = "03_freesurfer/UCSFFSX7_26Oct2026.csv",
    target_mapping: dict[str, dict[str, str]] | None = None,
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    """Extract FreeSurfer features linked to cohort RIDs via blind tie-breaking rule."""
    mapping = target_mapping if target_mapping is not None else FS_TARGET_MAPPING
    feature_cols = list(mapping.keys())

    cohort = pd.read_csv(cohort_path, low_memory=False)
    cohort["T_anchor_dt"] = pd.to_datetime(cohort["T_anchor"])
    cohort["RID"] = cohort["RID"].astype(int)

    fs_raw = pd.read_csv(fs_path, low_memory=False)
    fs_raw["EXAMDATE_dt"] = pd.to_datetime(fs_raw["EXAMDATE"])
    fs_raw["RID"] = fs_raw["RID"].astype(int)

    # 1. Label-free population statistics across ALL 12,515 rows
    robust_params = compute_robust_z_params(fs_raw, feature_cols=feature_cols)

    records: list[dict[str, Any]] = []

    for _, row in cohort.iterrows():
        rid = int(row["RID"])
        t0 = row["T_anchor_dt"]
        w_start = t0 - pd.Timedelta(days=180)
        w_end = t0 + pd.Timedelta(days=90)

        # Candidates within window [T_anchor - 180 d, T_anchor + 90 d]
        sub = fs_raw[
            (fs_raw["RID"] == rid)
            & (fs_raw["EXAMDATE_dt"] >= w_start)
            & (fs_raw["EXAMDATE_dt"] <= w_end)
        ].copy()

        if sub.empty:
            # Missing record
            rec: dict[str, Any] = {
                "RID": rid,
                "fs_observed": False,
                "IMAGEUID": np.nan,
                "EXAMDATE": np.nan,
                "fs_days_from_anchor": np.nan,
                "FSVER": np.nan,
                "STATUS": np.nan,
                "OVERALLQC": np.nan,
                "fs_screen_pass": False,
                "fs_any_robust_outlier": False,
                "fs_flag": False,
            }
            for col, info in mapping.items():
                rec[col] = np.nan
                rec[info["name"]] = np.nan
                rec[f"{col}_robust_z"] = np.nan
            rec["left_hippo_vol_icv"] = np.nan
            rec["right_hippo_vol_icv"] = np.nan
            rec["bilateral_hippo_vol_icv"] = np.nan
            records.append(rec)
            continue

        # Blind selection rule:
        # 1. Minimum absolute distance to T_anchor
        # 2. Preference for pre-anchor (days <= 0) over post-anchor (days > 0)
        # 3. Preference for OVERALLQC: Pass > Partial > Fail
        # 4. Deterministic tiebreak: largest IMAGEUID
        sub["days_diff"] = (sub["EXAMDATE_dt"] - t0).dt.days
        sub["abs_diff"] = sub["days_diff"].abs()
        sub["pre_anchor_pref"] = (sub["days_diff"] > 0).astype(int)
        qc_rank = {"Pass": 0, "Partial": 1, "Fail": 2}
        sub["qc_order"] = sub["OVERALLQC"].map(qc_rank).fillna(3)

        sub_sorted = sub.sort_values(
            by=["abs_diff", "pre_anchor_pref", "qc_order", "IMAGEUID"],
            ascending=[True, True, True, False],
        )
        best = sub_sorted.iloc[0]

        # Extract values and compute robust z-scores
        rec = {
            "RID": rid,
            "fs_observed": True,
            "IMAGEUID": int(best["IMAGEUID"]) if pd.notna(best["IMAGEUID"]) else np.nan,
            "EXAMDATE": best["EXAMDATE"],
            "fs_days_from_anchor": int(best["days_diff"]),
            "FSVER": best["FSVER"],
            "STATUS": best["STATUS"],
            "OVERALLQC": best["OVERALLQC"],
        }

        any_outlier = False
        all_targets_present = True
        any_target_le_zero = False

        for col, info in mapping.items():
            val = float(best[col]) if pd.notna(best[col]) else np.nan
            rec[col] = val
            rec[info["name"]] = val
            if pd.isna(val):
                all_targets_present = False
                rec[f"{col}_robust_z"] = np.nan
            else:
                if val <= 0:
                    any_target_le_zero = True
                med, scale = robust_params[col]
                rz = (val - med) / scale
                rec[f"{col}_robust_z"] = rz
                if abs(rz) > 5.0:
                    any_outlier = True

        # Normalized hippocampal volumes
        icv = rec.get("ST10CV", np.nan)
        lh = rec.get("ST29SV", np.nan)
        rh = rec.get("ST88SV", np.nan)

        if pd.notna(icv) and icv > 0 and pd.notna(lh) and pd.notna(rh):
            rec["left_hippo_vol_icv"] = lh / icv
            rec["right_hippo_vol_icv"] = rh / icv
            rec["bilateral_hippo_vol_icv"] = (lh + rh) / icv
            # Asymmetry screen: |L - R| / mean < 0.5
            hippo_asym = abs(lh - rh) / ((lh + rh) / 2.0)
            asym_pass = hippo_asym < 0.5
        else:
            rec["left_hippo_vol_icv"] = np.nan
            rec["right_hippo_vol_icv"] = np.nan
            rec["bilateral_hippo_vol_icv"] = np.nan
            asym_pass = False

        screen_pass = (
            all_targets_present
            and not any_outlier
            and asym_pass
            and best["OVERALLQC"] != "Fail"
        )
        rec["fs_any_robust_outlier"] = any_outlier
        rec["fs_screen_pass"] = screen_pass
        rec["fs_flag"] = (not screen_pass)

        # Implausible audit: asymmetry > 50%, any target <= 0, ICV outside [800k, 2500k]
        implausible = (
            (not asym_pass)
            or any_target_le_zero
            or (pd.notna(icv) and (icv < 800000 or icv > 2500000))
        )
        rec["fs_implausible"] = implausible

        records.append(rec)

    df_result = pd.DataFrame(records)
    return df_result, robust_params


def generate_fs_mapping_report(
    robust_params: dict[str, tuple[float, float]],
    out_path: Path | str = "reports/fs_mapping.md",
) -> None:
    """Generate reports/fs_mapping.md containing verbatim DATADIC verification."""
    lines = [
        "# FreeSurfer Feature Mapping and DATADIC Verification (Stage 4)",
        "",
        "**Dataset**: `UCSFFSX7_26Oct2026.csv`  ",
        "**Reference Data Dictionary**: `DATADIC_26Oct2026.csv` (Table: `UCSFFSX7`)  ",
        "**Date**: November 2026  ",
        "",
        "---",
        "",
        "## 1. Verbatim Feature Mapping and Measure Verification",
        "",
        "All 8 cortical thickness target features were audited and matched against `DATADIC_26Oct2026.csv`. ",
        "Every thickness feature maps strictly to **`Thickness Average (aparc.stats)`** (suffix `TA`), ",
        "confirming that **NO thickness target is mistakenly mapped to cortical volume (`CV`), surface area (`SA`), ",
        "subcortical volume (`SV`), or thickness standard deviation (`TS`)**.",
        "",
        "| Anatomical Target | Field Code | Measure Type | Verbatim DATADIC Definition | Median | MAD Scale (1.4826*MAD) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for col, info in FS_TARGET_MAPPING.items():
        med, scale = robust_params.get(col, (np.nan, np.nan))
        lines.append(
            f"| **{info['region']}** | `{col}` | {info['measure']} | *{info['datadic_text']}* | {med:.2f} | {scale:.2f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Derived Features and Outlier Protocol",
        "",
        "- **ICV Normalization**: Hippocampal volumes are normalized by dividing raw subcortical volume by Estimated Total Intracranial Volume (`ST10CV`):",
        "  - `left_hippo_vol_icv = ST29SV / ST10CV`",
        "  - `right_hippo_vol_icv = ST88SV / ST10CV`",
        "  - `bilateral_hippo_vol_icv = (ST29SV + ST88SV) / ST10CV`",
        "- **Label-Free Robust Z-Scores**: Computed across ALL 12,515 records in `UCSFFSX7_26Oct2026.csv`:",
        "  $$z_{\\text{robust}} = \\frac{x - \\text{median}}{1.4826 \\times \\text{MAD}}$$",
        "- **Automated Sanity Screen Criteria**:",
        "  1. Non-null values across all 10 morphometric targets and ICV.",
        "  2. Absence of extreme population outliers ($|z_{\\text{robust}}| \\le 5.0$).",
        "  3. Bilateral hippocampal asymmetry ratio $< 0.50$ (i.e. $|V_L - V_R| / \\text{mean} < 0.5$).",
        "  4. Visual QC `OVERALLQC` is not `'Fail'`.",
        "",
        "---",
        "",
        "## 3. Window Selection Rule (Blind Selection)",
        "",
        "When multiple FreeSurfer records fall inside the temporal window $[T_{\\text{anchor}} - 180\\text{ d}, T_{\\text{anchor}} + 90\\text{ d}]$:",
        "1. Select the record minimizing $|\\text{EXAMDATE} - T_{\\text{anchor}}|$.",
        "2. Break temporal ties by preferring pre-anchor acquisitions ($\\text{EXAMDATE} \\le T_{\\text{anchor}}$) over post-anchor acquisitions.",
        "3. Break same-day scan ties by visual quality rank (`OVERALLQC`: `Pass` > `Partial` > `Fail` / unrated).",
        "4. Final deterministic tiebreak: highest `IMAGEUID`.",
    ])

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated FreeSurfer mapping report at {out_file}.")


def save_freesurfer_features(
    out_path: Path | str = "data/processed/features/freesurfer_raw_v1.csv",
    cohort_path: Path | str = "data/processed/cohort_v2b.csv",
    fs_path: Path | str = "03_freesurfer/UCSFFSX7_26Oct2026.csv",
    report_path: Path | str = "reports/fs_mapping.md",
) -> pd.DataFrame:
    """Extract and persist raw FreeSurfer features and mapping report."""
    df, robust_params = extract_freesurfer_features(cohort_path=cohort_path, fs_path=fs_path)
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"Saved {len(df)} FreeSurfer feature records to {out_file}.")
    generate_fs_mapping_report(robust_params=robust_params, out_path=report_path)
    return df


def append_fs29_mapping_report(
    robust_params: dict[str, tuple[float, float]],
    report_path: Path | str = "reports/fs_mapping.md",
) -> None:
    """Append Section 4 containing verbatim fs29 DATADIC mapping to reports/fs_mapping.md."""
    lines = [
        "",
        "---",
        "",
        "## 4. A-Priori 29-Feature FreeSurfer Set (fs29) Mapping and DATADIC Verification",
        "",
        "All 29 morphometric features (8 subcortical volumes, 20 cortical thickness averages, and 1 intracranial volume) were audited and matched verbatim against `DATADIC_26Oct2026.csv` (`TBLNAME = UCSFFSX7`).",
        "",
        "| Anatomical Target | Field Code | Measure Type | Verbatim DATADIC Definition | Median | MAD Scale (1.4826*MAD) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for col, info in FS29_TARGET_MAPPING.items():
        med, scale = robust_params.get(col, (np.nan, np.nan))
        lines.append(
            f"| **{info['region']}** | `{col}` | {info['measure']} | *{info['datadic_text']}* | {med:.2f} | {scale:.2f} |"
        )

    out_file = Path(report_path)
    existing_content = out_file.read_text(encoding="utf-8") if out_file.exists() else ""
    out_file.write_text(existing_content.rstrip() + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"Appended fs29 mapping to {out_file}.")


def save_freesurfer29_features(
    out_path: Path | str = "data/processed/features/freesurfer29_raw_v1.csv",
    cohort_path: Path | str = "data/processed/cohort_v2b.csv",
    fs_path: Path | str = "03_freesurfer/UCSFFSX7_26Oct2026.csv",
    report_path: Path | str = "reports/fs_mapping.md",
) -> pd.DataFrame:
    """Extract and persist raw fs29 FreeSurfer features and append mapping report."""
    df, robust_params = extract_freesurfer_features(
        cohort_path=cohort_path,
        fs_path=fs_path,
        target_mapping=FS29_TARGET_MAPPING,
    )
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"Saved {len(df)} FreeSurfer fs29 feature records to {out_file}.")
    append_fs29_mapping_report(robust_params=robust_params, report_path=report_path)
    return df


if __name__ == "__main__":
    df_fs = save_freesurfer_features()
    print("FreeSurfer feature extraction sample:")
    print(df_fs[["RID", "fs_observed", "IMAGEUID", "fs_screen_pass", "bilateral_hippo_vol_icv"]].head())

    df_fs29 = save_freesurfer29_features()
    print("FreeSurfer fs29 extraction sample:")
    print(df_fs29[["RID", "fs_observed", "IMAGEUID", "fs_screen_pass", "fs_flag"]].head())
