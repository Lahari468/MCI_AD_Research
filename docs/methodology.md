# Methodology Overview

## Pipeline Architecture
The research workflow comprises five modular stages:

```
Raw ADNI Data (Clinical, FreeSurfer, MRI DICOM)
    │
    ▼
[1. Cohort Construction & Verification]
    ├── Age, diagnosis, 730-day progression horizon
    └── Stratified 3-Repeat × 5-Fold split generation
    │
    ▼
[2. Feature Engineering & Preprocessing]
    ├── Clinical: 10 core demographic and cognitive features
    ├── FreeSurfer: 29 regional volumes ICV-normalized (ST10CV)
    └── 3D MRI: N4 bias correction, 12 DOF FLIRT registration to MNI152 2mm
    │
    ▼
[3. Predictive Modeling]
    ├── Model A: Clinical 10 + FreeSurfer 29
    ├── Model B (Stage 12): Clinical 10 + FreeSurfer 29 + Whole-Volume 3D CNN
    ├── Model B (Stage 19): Clinical 10 + FreeSurfer 29 + 16 Subcortical ROIs
    └── Comparators: Clinical Only, MRI Only, Non-linear Gradient Boosting
    │
    ▼
[4. Out-of-Fold Evaluation]
    ├── Participant-level repeat-averaged probability aggregation
    └── Discrimination (AUROC, PR-AUC), Calibration (Brier Score)
    │
    ▼
[5. Statistical Inference]
    └── Participant-level paired bootstrap (B = 2,000 resamples, seed 42)
```

## Prevention of Data Leakage
- **Strict Partitioning**: Outer validation folds are held out completely.
- **Nested Tuning**: Hyperparameter $C$ tuning and operating threshold selection ($t_{\text{youden}}$) are performed strictly within a nested 15% inner validation split.
- **Inner Imputation & Scaling**: Feature medians and standard deviations are computed strictly on the training partition and applied forward.
