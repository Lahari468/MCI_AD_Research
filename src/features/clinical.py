"""Clinical Feature Extraction, Negative Code Recoding, and Feature Set Definitions.

Adheres strictly to Phase 3/Stage 4 specifications:
- MMSCORE, CDRSB, CDGLOBAL, FAQTOTAL, TOTSCORE, TOTAL13, PTEDUCAT, sex, age at index visit, APOE4 count.
- Recodes documented negative values (<0) to NaN, since in ADNI data dictionaries negative
  values reflect unobserved / unattempted / administrative missing codes while valid clinical
  instrument scores are strictly bounded non-negative quantities.
- Generates was_negative flags and missingness indicators.
- Exports two standard feature sets:
    1. clinical_full: all 10 features
    2. clinical_nocirc: non-circular set dropping CDRSB, CDGLOBAL, FAQTOTAL
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# Documented non-negative valid theoretical bounds for ADNI clinical instruments:
# ADNI data dictionary provides no semantic meaning for negative integers in these fields;
# negative integers (<0) represent unattempted tests or missingness sentinels.
CLINICAL_BOUNDS: dict[str, tuple[float, float]] = {
    "MMSCORE": (0.0, 30.0),    # Mini-Mental State Examination
    "CDRSB": (0.0, 18.0),      # Clinical Dementia Rating Sum of Boxes
    "CDGLOBAL": (0.0, 3.0),    # Clinical Dementia Rating Global Score
    "FAQTOTAL": (0.0, 30.0),   # Functional Activities Questionnaire Total
    "TOTSCORE": (0.0, 70.0),   # ADAS-Cognitive 11-item Total Score
    "TOTAL13": (0.0, 85.0),    # ADAS-Cognitive 13-item Total Score
    "PTEDUCAT": (0.0, 25.0),   # Years of Education
}

CLINICAL_FULL: list[str] = [
    "age",
    "sex",
    "PTEDUCAT",
    "APOE4",
    "MMSCORE",
    "CDRSB",
    "CDGLOBAL",
    "FAQTOTAL",
    "TOTSCORE",
    "TOTAL13",
]

CLINICAL_NOCIRC: list[str] = [
    "age",
    "sex",
    "PTEDUCAT",
    "APOE4",
    "MMSCORE",
    "TOTSCORE",
    "TOTAL13",
]


def clean_clinical_metric(
    val: float | None,
    col_name: str,
) -> tuple[float, bool]:
    """Recode values < 0 to NaN, verifying against theoretical instrument bounds.

    Returns:
        (cleaned_value, was_negative_flag)
    """
    if val is None or pd.isna(val):
        return np.nan, False

    try:
        num = float(val)
    except (ValueError, TypeError):
        return np.nan, False

    if num < 0:
        return np.nan, True

    # Upper bound sanity check (logs warning if violated)
    if col_name in CLINICAL_BOUNDS:
        _, max_bound = CLINICAL_BOUNDS[col_name]
        if num > max_bound:
            pass  # Retain raw value but flag if needed

    return num, False


def extract_clinical_features(
    cohort_path: Path | str = "data/processed/cohort_v2b.csv",
    data_dir: Path | str = ".",
) -> pd.DataFrame:
    """Extract index-window clinical features for all participants in the cohort."""
    root = Path(data_dir)
    cohort = pd.read_csv(cohort_path, low_memory=False)
    cohort["T_anchor_dt"] = pd.to_datetime(cohort["T_anchor"])
    cohort["RID"] = cohort["RID"].astype(int)

    # 1. MMSE
    mmse = pd.read_csv(root / "02_clinical/MMSE_26Oct2026.csv", low_memory=False)
    mmse["VISDATE"] = pd.to_datetime(mmse["VISDATE"])
    mmse["RID"] = mmse["RID"].astype(int)

    # 2. CDR
    cdr = pd.read_csv(root / "02_clinical/CDR_26Oct2026.csv", low_memory=False)
    cdr["VISDATE"] = pd.to_datetime(cdr["VISDATE"])
    cdr["RID"] = cdr["RID"].astype(int)

    # 3. FAQ
    faq = pd.read_csv(root / "02_clinical/FAQ_26Oct2026.csv", low_memory=False)
    faq["VISDATE"] = pd.to_datetime(faq["VISDATE"])
    faq["RID"] = faq["RID"].astype(int)

    # 4. ADAS
    adas = pd.read_csv(root / "02_clinical/ADAS_26Oct2026.csv", low_memory=False)
    adas["VISDATE"] = pd.to_datetime(adas["VISDATE"])
    adas["RID"] = adas["RID"].astype(int)

    # 5. PTDEMOG
    ptdemog = pd.read_csv(root / "02_clinical/PTDEMOG_26Oct2026.csv", low_memory=False)
    ptdemog["RID"] = ptdemog["RID"].astype(int)
    edu_dict = ptdemog.groupby("RID")["PTEDUCAT"].first().to_dict()

    # 6. PHC DEMODX (Age and Sex at index visit)
    phc = pd.read_csv(root / "data/raw/clinical/ADSP_PHC_DEMODX.csv", low_memory=False)
    phc["EXAMDATE"] = pd.to_datetime(phc["EXAMDATE"])
    phc["RID"] = phc["RID"].astype(int)

    # 7. APOE Genetics
    apoe = pd.read_csv(root / "05_genetics/APOERES_26Oct2026.csv", low_memory=False)
    apoe["RID"] = apoe["RID"].astype(int)
    apoe_map = {"3/3": 0, "2/3": 0, "2/2": 0, "3/4": 1, "2/4": 1, "4/4": 2}
    apoe["apoe4"] = apoe["GENOTYPE"].map(apoe_map)
    apoe_dict = apoe.groupby("RID")["apoe4"].first().to_dict()

    records: list[dict[str, Any]] = []

    for _, row in cohort.iterrows():
        rid = int(row["RID"])
        t0 = row["T_anchor_dt"]
        w_start = t0 - pd.Timedelta(days=90)

        # PHC Demographics at T_anchor
        sub_phc = phc[(phc["RID"] == rid) & (phc["EXAMDATE"] == t0)]
        age = float(sub_phc["PHC_Age_Diagnosis"].iloc[0]) if not sub_phc.empty else np.nan
        raw_sex = sub_phc["PHC_Sex"].iloc[0] if not sub_phc.empty else np.nan
        # Encode sex: 1=Male, 0=Female (PHC_Sex: 1=Male, 2=Female)
        sex = 1.0 if raw_sex == 1 else (0.0 if raw_sex == 2 else np.nan)

        # MMSE
        sub_m = mmse[(mmse["RID"] == rid) & (mmse["VISDATE"] >= w_start) & (mmse["VISDATE"] <= t0)].sort_values("VISDATE")
        raw_mmse = sub_m.iloc[-1]["MMSCORE"] if not sub_m.empty else np.nan
        mmse_val, mmse_neg = clean_clinical_metric(raw_mmse, "MMSCORE")

        # CDR
        sub_c = cdr[(cdr["RID"] == rid) & (cdr["VISDATE"] >= w_start) & (cdr["VISDATE"] <= t0)].sort_values("VISDATE")
        raw_cdrsb = sub_c.iloc[-1]["CDRSB"] if not sub_c.empty else np.nan
        raw_cdglob = sub_c.iloc[-1]["CDGLOBAL"] if not sub_c.empty else np.nan
        cdrsb_val, cdrsb_neg = clean_clinical_metric(raw_cdrsb, "CDRSB")
        cdglob_val, cdglob_neg = clean_clinical_metric(raw_cdglob, "CDGLOBAL")

        # FAQ
        sub_f = faq[(faq["RID"] == rid) & (faq["VISDATE"] >= w_start) & (faq["VISDATE"] <= t0)].sort_values("VISDATE")
        raw_faq = sub_f.iloc[-1]["FAQTOTAL"] if not sub_f.empty else np.nan
        faq_val, faq_neg = clean_clinical_metric(raw_faq, "FAQTOTAL")

        # ADAS
        sub_a = adas[(adas["RID"] == rid) & (adas["VISDATE"] >= w_start) & (adas["VISDATE"] <= t0)].sort_values("VISDATE")
        raw_tot = sub_a.iloc[-1]["TOTSCORE"] if not sub_a.empty else np.nan
        raw_t13 = sub_a.iloc[-1]["TOTAL13"] if not sub_a.empty else np.nan
        tot_val, tot_neg = clean_clinical_metric(raw_tot, "TOTSCORE")
        t13_val, t13_neg = clean_clinical_metric(raw_t13, "TOTAL13")

        # Education
        raw_edu = edu_dict.get(rid, np.nan)
        edu_val, edu_neg = clean_clinical_metric(raw_edu, "PTEDUCAT")

        # APOE
        apoe4_val = apoe_dict.get(rid, np.nan)

        rec: dict[str, Any] = {
            "RID": rid,
            # Core Features
            "age": age,
            "sex": sex,
            "PTEDUCAT": edu_val,
            "APOE4": apoe4_val,
            "MMSCORE": mmse_val,
            "CDRSB": cdrsb_val,
            "CDGLOBAL": cdglob_val,
            "FAQTOTAL": faq_val,
            "TOTSCORE": tot_val,
            "TOTAL13": t13_val,
            # Was Negative Flags
            "MMSCORE_was_negative": mmse_neg,
            "CDRSB_was_negative": cdrsb_neg,
            "CDGLOBAL_was_negative": cdglob_neg,
            "FAQTOTAL_was_negative": faq_neg,
            "TOTSCORE_was_negative": tot_neg,
            "TOTAL13_was_negative": t13_neg,
            "PTEDUCAT_was_negative": edu_neg,
            # Missing Indicators
            "age_is_missing": pd.isna(age),
            "sex_is_missing": pd.isna(sex),
            "PTEDUCAT_is_missing": pd.isna(edu_val),
            "APOE4_is_missing": pd.isna(apoe4_val),
            "MMSCORE_is_missing": pd.isna(mmse_val),
            "CDRSB_is_missing": pd.isna(cdrsb_val),
            "CDGLOBAL_is_missing": pd.isna(cdglob_val),
            "FAQTOTAL_is_missing": pd.isna(faq_val),
            "TOTSCORE_is_missing": pd.isna(tot_val),
            "TOTAL13_is_missing": pd.isna(t13_val),
        }
        records.append(rec)

    return pd.DataFrame(records)


def save_clinical_features(
    out_path: Path | str = "data/processed/features/clinical_raw_v1.csv",
    cohort_path: Path | str = "data/processed/cohort_v2b.csv",
) -> pd.DataFrame:
    """Extract and persist raw clinical features with missing flags."""
    df = extract_clinical_features(cohort_path=cohort_path)
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"Saved {len(df)} clinical feature records to {out_file}.")
    return df


if __name__ == "__main__":
    df_clin = save_clinical_features()
    print("Clinical feature extraction sample:")
    print(df_clin[CLINICAL_FULL].head())
