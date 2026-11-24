#!/usr/bin/env python3
"""
Atlas-Based Anatomical Subcortical ROI Feature Extraction Pipeline
Extracts intensity and contrast features from MNI152 2mm preprocessed baseline T1 MRI volumes
using the Harvard-Oxford Subcortical Structural Atlas.

Target Cohort: Verified ADNI development cohort participants (N=564 with usable baseline MRI).
Reference Atlas: Harvard-Oxford Subcortical Structural Atlas (maxprob-thr25-2mm).
Output File: data/processed/features/mri_subcortical_roi_features.csv
"""

import os, sys, time, json
import numpy as np
import pandas as pd
import nibabel as nib

# -------------------------------------------------------------
# 0. Paths and Configuration
# -------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(PROJECT_ROOT)

ATLAS_PATH = 'data/reference/atlases/HarvardOxford-sub-maxprob-thr25-2mm.nii.gz'
BRAIN_MASK_PATH = 'data/reference/atlases/MNI152_T1_2mm_brain_mask.nii.gz'
MRI_DIR = 'data/interim/mri_mni2mm_v2'
MANIFEST_PATH = 'data/processed/mri/manifests/mri_manifest_713.csv'
SPLITS_PATH = 'data/processed/splits/splits_v1.csv'
COHORT_PATH = 'data/processed/cohort_v2b.csv'
OUT_CSV = 'data/processed/features/mri_subcortical_roi_features.csv'

# -------------------------------------------------------------
# 1. Verify Inputs and Cohort Isolation
# -------------------------------------------------------------
assert os.path.exists(ATLAS_PATH), f"Atlas not found: {ATLAS_PATH}"
assert os.path.exists(BRAIN_MASK_PATH), f"Brain mask not found: {BRAIN_MASK_PATH}"
assert os.path.exists(MRI_DIR), f"MRI directory not found: {MRI_DIR}"

sp = pd.read_csv(SPLITS_PATH)
test_rids = set(sp[sp.split == 'test']['rid'].unique())
dev_rids = set(sp[sp.split != 'test']['rid'].unique())
assert len(test_rids) == 156, f"Expected 156 test RIDs, got {len(test_rids)}"
assert len(dev_rids) == 622, f"Expected 622 dev RIDs, got {len(dev_rids)}"

mf = pd.read_csv(MANIFEST_PATH)
coh = pd.read_csv(COHORT_PATH)

# Identify development cohort with usable MRI on disk
usable_dev_mf = mf[mf.rid.isin(dev_rids) & mf.mri_usable & mf.tensor_present].copy()
usable_dev_rids = sorted(usable_dev_mf['rid'].unique())
assert len(usable_dev_rids) == 564, f"Expected 564 usable dev participants, got {len(usable_dev_rids)}"
assert len(set(usable_dev_rids) & test_rids) == 0, "CRITICAL: Test participant detected in extraction cohort!"

print(f"[{time.strftime('%X')}] Verified isolation: {len(usable_dev_rids)} development participants identified.")

# -------------------------------------------------------------
# 2. Atlas Definition & Voxel Verification
# -------------------------------------------------------------
atlas_img = nib.load(ATLAS_PATH)
atlas_data = atlas_img.get_fdata().astype(int)
mask_img = nib.load(BRAIN_MASK_PATH)
mask_data = mask_img.get_fdata() > 0

# Subcortical Atlas Label Map (Harvard-Oxford NIfTI label values)
ROI_LABELS = {
    # Reference Tissue
    'wm_left': 1,
    'wm_right': 12,
    # Primary AD-Vulnerable Targets
    'hippo_left': 9,
    'hippo_right': 19,
    'amygdala_left': 10,
    'amygdala_right': 20,
    'latvent_left': 3,
    'latvent_right': 14,
    # Anatomical Control Structures
    'thalamus_left': 4,
    'thalamus_right': 15,
    'caudate_left': 5,
    'caudate_right': 16,
    'putamen_left': 6,
    'putamen_right': 17
}

# Combined white matter mask for intensity calibration
wm_mask = (atlas_data == ROI_LABELS['wm_left']) | (atlas_data == ROI_LABELS['wm_right'])
assert np.sum(wm_mask) > 50000, "White matter mask too small!"

# -------------------------------------------------------------
# 3. Feature Extraction Loop
# -------------------------------------------------------------
records = []
t0 = time.time()

for rid in usable_dev_rids:
    tensor_path = os.path.join(MRI_DIR, f"{rid}.npy")
    if not os.path.exists(tensor_path):
        records.append({
            'rid': rid,
            'image_id': int(usable_dev_mf[usable_dev_mf.rid == rid]['image_id'].iloc[0]),
            'extraction_status': 'FILE_NOT_FOUND',
            'qc_flag': 'MISSING_TENSOR'
        })
        continue

    # Load 3D tensor: shape (1, 91, 109, 91), float16 -> float32
    raw_tensor = np.load(tensor_path)
    vol = raw_tensor[0].astype(np.float32)

    # QC check on tensor integrity
    if np.isnan(vol).any() or np.isinf(vol).any():
        records.append({
            'rid': rid,
            'image_id': int(usable_dev_mf[usable_dev_mf.rid == rid]['image_id'].iloc[0]),
            'extraction_status': 'FAIL_NAN_INF',
            'qc_flag': 'NUMERICAL_INVALID'
        })
        continue

    # Reference WM median intensity
    wm_vals = vol[wm_mask]
    wm_med = float(np.median(wm_vals))

    row = {
        'rid': int(rid),
        'image_id': int(usable_dev_mf[usable_dev_mf.rid == rid]['image_id'].iloc[0]),
        'wm_median': round(wm_med, 4),
        'extraction_status': 'OK',
        'qc_flag': 'CLEAN'
    }

    # Extract features for each structure
    extracted_medians = {}
    extracted_means = {}

    for roi_name, lbl_idx in ROI_LABELS.items():
        if roi_name in ['wm_left', 'wm_right']:
            continue
        roi_voxels = vol[atlas_data == lbl_idx]
        v_count = len(roi_voxels)
        
        # Summary statistics relative to WM median
        m_mean = float(np.mean(roi_voxels) - wm_med)
        m_med = float(np.median(roi_voxels) - wm_med)
        m_p10 = float(np.percentile(roi_voxels, 10) - wm_med)
        m_p90 = float(np.percentile(roi_voxels, 90) - wm_med)
        m_std = float(np.std(roi_voxels))
        m_iqr = float(np.percentile(roi_voxels, 75) - np.percentile(roi_voxels, 25))

        row[f"{roi_name}_mean"] = round(m_mean, 4)
        row[f"{roi_name}_median"] = round(m_med, 4)
        row[f"{roi_name}_p10"] = round(m_p10, 4)
        row[f"{roi_name}_p90"] = round(m_p90, 4)
        row[f"{roi_name}_std"] = round(m_std, 4)
        row[f"{roi_name}_iqr"] = round(m_iqr, 4)
        row[f"{roi_name}_nvox"] = int(v_count)

        extracted_medians[roi_name] = m_med
        extracted_means[roi_name] = m_mean

    # Bilateral & Composite Biomarkers
    # Bilateral Hippocampus contrast
    row['hippo_bilateral_median'] = round((extracted_medians['hippo_left'] + extracted_medians['hippo_right']) / 2.0, 4)
    row['hippo_bilateral_mean'] = round((extracted_means['hippo_left'] + extracted_means['hippo_right']) / 2.0, 4)
    # Bilateral Amygdala contrast
    row['amygdala_bilateral_median'] = round((extracted_medians['amygdala_left'] + extracted_medians['amygdala_right']) / 2.0, 4)
    # Bilateral Lateral Ventricle contrast
    row['latvent_bilateral_median'] = round((extracted_medians['latvent_left'] + extracted_medians['latvent_right']) / 2.0, 4)
    # Ventricle-to-Hippocampus Difference (Hallmark contrast of medial temporal atrophy / ex vacuo dilation)
    row['vent_to_hippo_diff'] = round(row['latvent_bilateral_median'] - row['hippo_bilateral_median'], 4)

    records.append(row)

df_out = pd.DataFrame(records)
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
df_out.to_csv(OUT_CSV, index=False)

print(f"[{time.strftime('%X')}] Extraction complete in {time.time()-t0:.2f}s.")
print(f"Output saved to: {OUT_CSV}")
print(f"Shape: {df_out.shape} ({len(df_out)} participants, {df_out.shape[1]} columns)")
print(f"Status counts:\n{df_out['extraction_status'].value_counts()}")
