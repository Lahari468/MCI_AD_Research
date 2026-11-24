"""Participant-level Splitting Pipeline and Secure Split Loader.
Supports:
  - Stratified 20% locked test set partitioned by outcome label x ADNI phase group.
  - Stratified 5-fold cross-validation with 3 repeats on the development set.
  - Guarded access to locked test set (requires FINAL_EVAL=1 environment variable).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
import yaml


def load_config(config_path: Path | str = "configs/base.yaml") -> dict[str, Any]:
    """Load configuration YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def group_phase(phase: str) -> str:
    """Group ADNI phases to avoid thin strata."""
    if phase == "ADNI1":
        return "ADNI1"
    elif phase in ["ADNIGO", "ADNI2"]:
        return "ADNIGO+ADNI2"
    elif phase in ["ADNI3", "ADNI4"]:
        return "ADNI3+ADNI4"
    return "OTHER"


def generate_splits(
    cohort_path: Path | str = "data/processed/cohort_v2b.csv",
    config_path: Path | str = "configs/base.yaml",
    manifest_mri_path: Path | str = "04_mri_metadata/T1w_Structural_MRI_Cohort_Manifest_26Oct2026.csv",
    out_dir: Path | str = "data/processed/splits",
) -> dict[str, Any]:
    """Generate locked test and 3x5-fold CV splits for the eligible cohort."""
    cfg = load_config(config_path)
    seed = cfg.get("system", {}).get("seed", 42)

    cohort = pd.read_csv(cohort_path, low_memory=False)
    elig = cohort[cohort["exclusion_reason"] == "ELIGIBLE"].copy()
    elig["RID"] = elig["RID"].astype(int)

    # Stratification feature: label x phase_group
    elig["phase_grp"] = elig["adni_phase"].apply(group_phase)
    elig["strata"] = elig["label_primary"].astype(int).astype(str) + "_" + elig["phase_grp"]

    # MRI in manifest indicator
    mri_man = pd.read_csv(manifest_mri_path, low_memory=False)
    mri_man["image_date"] = pd.to_datetime(mri_man["image_date"])
    elig["T_anchor_dt"] = pd.to_datetime(elig["T_anchor"])

    has_mri = []
    for _, r in elig.iterrows():
        rid = r["RID"]
        t0 = r["T_anchor_dt"]
        sub = mri_man[
            (mri_man["subject_id"].str.contains(f"_{rid:04d}$") | mri_man["subject_id"].str.contains(f"_{rid}$"))
            & (mri_man["image_date"] >= t0 - pd.Timedelta(days=180))
            & (mri_man["image_date"] <= t0 + pd.Timedelta(days=90))
        ]
        has_mri.append(not sub.empty)
    elig["has_mri_manifest"] = has_mri

    # 1. Locked test split: 20% stratified by label x phase_group
    dev_df, test_df = train_test_split(
        elig,
        test_size=0.20,
        random_state=seed,
        stratify=elig["strata"],
    )

    records: list[dict[str, Any]] = []

    # Record locked test
    for rid in test_df["RID"]:
        records.append({"rid": int(rid), "split": "test", "repeat": -1, "fold": -1})

    # 2. Dev set 5-fold CV x 3 repeats
    dev_rids = dev_df["RID"].values
    dev_strata = dev_df["strata"].values

    fold_metrics = []
    for rep in range(3):
        rep_seed = seed + rep * 100
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rep_seed)
        for fold, (train_idx, val_idx) in enumerate(skf.split(dev_rids, dev_strata)):
            val_df_fold = dev_df.iloc[val_idx]
            val_prev = float(val_df_fold["label_primary"].mean())
            fold_metrics.append({
                "repeat": rep,
                "fold": fold,
                "n_val": len(val_df_fold),
                "prevalence": val_prev,
            })
            for rid in val_df_fold["RID"]:
                records.append({"rid": int(rid), "split": "dev", "repeat": rep, "fold": fold})

    splits_df = pd.DataFrame(records)
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    splits_csv = out_dir_path / "splits_v1.csv"
    splits_df.to_csv(splits_csv, index=False)

    # Compute SHA256 of splits_v1.csv
    with open(splits_csv, "rb") as f:
        sha256_hash = hashlib.sha256(f.read()).hexdigest()

    manifest = {
        "seed": seed,
        "n_eligible": len(elig),
        "n_test": len(test_df),
        "n_dev": len(dev_df),
        "test_prevalence": float(test_df["label_primary"].mean()),
        "dev_prevalence": float(dev_df["label_primary"].mean()),
        "test_mri_manifest_share": float(test_df["has_mri_manifest"].mean()),
        "dev_mri_manifest_share": float(dev_df["has_mri_manifest"].mean()),
        "fold_metrics": fold_metrics,
        "splits_csv": str(splits_csv),
        "sha256": sha256_hash,
    }

    manifest_json = out_dir_path / "splits_manifest.json"
    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def load_dev_fold(
    repeat: int,
    fold: int,
    splits_path: Path | str = "data/processed/splits/splits_v1.csv",
) -> tuple[list[int], list[int]]:
    """Load train and validation RIDs for a specific CV repeat and fold."""
    df = pd.read_csv(splits_path)
    dev_rep = df[(df["split"] == "dev") & (df["repeat"] == repeat)]
    if dev_rep.empty:
        raise ValueError(f"No records found for dev split repeat={repeat}")

    val_rids = dev_rep[dev_rep["fold"] == fold]["rid"].tolist()
    train_rids = dev_rep[dev_rep["fold"] != fold]["rid"].tolist()

    if not val_rids or not train_rids:
        raise ValueError(f"Invalid fold={fold} for repeat={repeat}")

    return train_rids, val_rids


def load_locked_test(
    splits_path: Path | str = "data/processed/splits/splits_v1.csv",
) -> list[int]:
    """Load locked test RIDs.

    Raises PermissionError unless environment variable FINAL_EVAL=1 is set.
    """
    if os.environ.get("FINAL_EVAL") != "1":
        raise PermissionError(
            "Access to locked test set is prohibited during development. "
            "Set environment variable FINAL_EVAL=1 to unlock."
        )

    df = pd.read_csv(splits_path)
    test_rids = df[df["split"] == "test"]["rid"].unique().tolist()
    return test_rids


if __name__ == "__main__":
    m = generate_splits()
    print("Splits generated successfully:")
    print(f"  SHA256: {m['sha256']}")
    print(f"  Test N: {m['n_test']} (prev: {m['test_prevalence']:.4f})")
    print(f"  Dev N: {m['n_dev']} (prev: {m['dev_prevalence']:.4f})")
