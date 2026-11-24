"""
Subcortical MRI ROI Feature Extraction
Module: src/features/mri_rois.py
Extracts intensity contrast and heterogeneity features from standardized MNI 2mm MRI volumes
using the Harvard-Oxford Subcortical Structural Atlas.
"""

import numpy as np

ROI_LABELS = {
    'wm_left': 1,
    'wm_right': 12,
    'hippo_left': 9,
    'hippo_right': 19,
    'amygdala_left': 10,
    'amygdala_right': 20,
    'latvent_left': 3,
    'latvent_right': 14,
    'thalamus_left': 4,
    'thalamus_right': 15,
    'caudate_left': 5,
    'caudate_right': 16,
    'putamen_left': 6,
    'putamen_right': 17
}


def extract_roi_statistics_from_volume(vol: np.ndarray, atlas_data: np.ndarray) -> dict:
    """
    vol: 3D numpy array of shape (91, 109, 91)
    atlas_data: 3D integer array of shape (91, 109, 91) matching Harvard-Oxford labels.
    """
    wm_mask = (atlas_data == ROI_LABELS['wm_left']) | (atlas_data == ROI_LABELS['wm_right'])
    wm_vals = vol[wm_mask]
    wm_med = float(np.median(wm_vals))

    record = {'wm_median': round(wm_med, 4)}
    extracted_medians = {}

    for roi_name, lbl_idx in ROI_LABELS.items():
        if roi_name in ['wm_left', 'wm_right']:
            continue
        roi_voxels = vol[atlas_data == lbl_idx]
        m_mean = float(np.mean(roi_voxels) - wm_med)
        m_med = float(np.median(roi_voxels) - wm_med)
        m_p10 = float(np.percentile(roi_voxels, 10) - wm_med)
        m_p90 = float(np.percentile(roi_voxels, 90) - wm_med)
        m_std = float(np.std(roi_voxels))
        m_iqr = float(np.percentile(roi_voxels, 75) - np.percentile(roi_voxels, 25))

        record[f"{roi_name}_mean"] = round(m_mean, 4)
        record[f"{roi_name}_median"] = round(m_med, 4)
        record[f"{roi_name}_p10"] = round(m_p10, 4)
        record[f"{roi_name}_p90"] = round(m_p90, 4)
        record[f"{roi_name}_std"] = round(m_std, 4)
        record[f"{roi_name}_iqr"] = round(m_iqr, 4)
        record[f"{roi_name}_nvox"] = int(len(roi_voxels))
        extracted_medians[roi_name] = m_med

    # Composite Features
    record['hippo_bilateral_median'] = round((extracted_medians['hippo_left'] + extracted_medians['hippo_right']) / 2.0, 4)
    record['amygdala_bilateral_median'] = round((extracted_medians['amygdala_left'] + extracted_medians['amygdala_right']) / 2.0, 4)
    record['latvent_bilateral_median'] = round((extracted_medians['latvent_left'] + extracted_medians['latvent_right']) / 2.0, 4)
    record['vent_to_hippo_diff'] = round(record['latvent_bilateral_median'] - record['hippo_bilateral_median'], 4)

    return record
