# STAGE 19 FINAL FORENSIC AUDIT REPORT
**Target Experiment**: Stage 19 Anatomically Informed Subcortical MRI ROI Incremental-Value Experiment  
**Target Repository**: `.`  
**Target Output Directory**: `results/mri_roi_incremental/`  
**Audit Date**: November 24, 2026  
**Auditor**: Antigravity Autonomous Research Assistant  
**Evaluation Protocol**: Strictly READ-ONLY Forensic Inspection (No retraining, no holdout evaluations, no script modifications)

---

## 1. EXECUTIVE VERDICT

### Overall Recommendation: **ACCEPT WITH CORRECTIONS**

Stage 19 represents an empirical and technical investigation into whether baseline 3D structural MRI contains incremental predictive value beyond clinical cognitive scales and FreeSurfer morphometry for 24-month MCI-to-AD progression.

The experimental implementation, cross-validation protocol, participant-level paired bootstrap evaluation, and reported numerical results were audited against the repository codebase, feature tables, and prediction files.

#### Key Audit Findings:
1. **Reproducibility (VERIFIED)**: Every reported performance metric in `results/mri_roi_incremental/repeat_summary.csv` and `results/mri_roi_incremental/paired_bootstrap_deltas.csv` is independently reproducible to the fourth decimal place ($0.0000$ discrepancy).
2. **Data & Cohort Hygiene (VERIFIED)**: Strictly $N = 564$ development participants with usable baseline MRI were evaluated. The locked holdout ($N = 156$) remained completely isolated ($0$ participants leaked). Exactly identical 3-repeat $\times$ 5-fold partitions were shared across all compared models.
3. **Leak-Free Preprocessing (VERIFIED)**: Median imputation and z-score standardization were fitted strictly on inner training folds.
4. **Methodological & Presentation Issues Identified (ISSUES FOUND)**:
   - **Feature Selection Discrepancy**: The extracted feature table contains 94 columns, but the primary target model (Model B) uses 16 features. While the 16 features were pre-specified based on prior neuroanatomical literature without label leakage, the report omitted a formal feature mapping table explaining why the remaining 78 columns were unused.
   - **Spatial Alignment Limitations**: The Harvard-Oxford atlas in MNI152 2 mm space was applied to linear affine registered volumes (FLIRT 12 DOF). While global alignment is verified, local non-linear warping was not performed.
   - **Overstated Scanner Normalization Claims**: The report claimed white matter median subtraction "eliminates inter-subject scanner gain drift and residual coil sensitivity variations." In non-quantitative T1 MRI across 1.5T/3.0T multi-site ADNI cohorts, white matter referencing provides an intra-subject baseline shift but cannot eliminate sequence-dependent contrast variations.
   - **Inner Validation Dual-Use**: The 15% inner validation split was used for both hyperparameter $C$ tuning and Youden $J$ threshold selection, introducing minor operating threshold noise.
   - **Subgroup Instability**: ADNI3 ($N_{\text{pos}} = 10$) and ADNIGO ($N_{\text{pos}} = 3$) have very low converter counts; their phase-level AUROCs are unstable and must not be cited as evidence of an MRI benefit.
   - **Overstated Scientific Claims**: Assertions that FreeSurfer "completely captures" MRI predictive information or that baseline MRI has "no biological value" are scientifically invalid and unsupported.

---

## 2. FEATURE EXTRACTION AUDIT (TASK 1)

### 2.1 Reference Atlas & Spatial Alignment

| Audit Parameter | Verified Repository Setting | Finding Status | Evidence / Source |
| :--- | :--- | :---: | :--- |
| **Atlas Name & Version** | Harvard-Oxford Subcortical Structural Atlas (`maxprob-thr25-2mm`) | **VERIFIED** | `data/reference/atlases/HarvardOxford-sub-maxprob-thr25-2mm.nii.gz` |
| **Spatial Resolution** | 2.0 mm isotropic voxel grid | **VERIFIED** | NIfTI header zooms: `(2.0, 2.0, 2.0)` mm |
| **Grid Dimensions** | $91 \times 109 \times 91$ voxels | **VERIFIED** | `img.shape == (91, 109, 91)` |
| **Affine Coordinate Matrix** | Standard FSL MNI152 orientation: $\text{diag}(-2, 2, 2, 1)$ with origin shift $[90, -126, -72]$ | **VERIFIED** | Match with standard template affine in `nibabel` |
| **Orientation Convention** | RPI (Right-to-Left, Posterior-to-Anterior, Inferior-to-Superior) | **VERIFIED** | FSL standard convention confirmed |
| **Input Tensor Concordance** | Shape $(1, 91, 109, 91)$ in `data/interim/mri_mni2mm_v2/<RID>.npy` | **VERIFIED** | Exact dimension and bounding-box match |
| **Registration Mechanism** | FLIRT 12 DOF affine registration to `MNI152_T1_2mm.nii.gz` with N4 bias correction | **VERIFIED** | `scripts/mri_preprocess_v2.py#L26-35` |
| **Local Non-linear Alignment** | Non-linear deformable registration (e.g., FNIRT / ANTs SyN) was **NOT** performed | **ISSUE FOUND** | Affine registration (12 DOF) cannot account for subject-specific local ventricular expansion and medial temporal cortical thinning. Standard space masks will partially sample CSF or white matter at regional boundaries. |
| **Visual Overlay QC Evidence** | 712 tri-planar slice overlay PNGs in `results/mri_v2/qc/<RID>.png` | **VERIFIED** | `results/mri_v2/qc/` ($N = 712$ images), numerical metrics in `results/mri_v2/preproc_log.csv` |
| **Atlas Contour Overlay QC** | Per-subject contour overlays of the subcortical atlas on individual tensors | **ISSUE FOUND** | Visual contour overlay of the atlas masks on individual patient volumes was not saved to disk. |

### 2.2 Subcortical Structures and Voxel Counts

The Harvard-Oxford atlas defines non-zero subcortical regions verified in `extract_mri_subcortical_rois.py#L63-82`:

| Region Index | Anatomical Structure | Hemisphere | Target Role | Voxel Count ($2\text{ mm}$) | Volume ($\text{cm}^3$) | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | Cerebral White Matter | Left | Intensity Calibration Reference | 31,371 | 250.97 | **VERIFIED** |
| **12** | Cerebral White Matter | Right | Intensity Calibration Reference | 31,361 | 250.89 | **VERIFIED** |
| **9** | Hippocampus | Left | Primary AD Target | 691 | 5.53 | **VERIFIED** |
| **19** | Hippocampus | Right | Primary AD Target | 700 | 5.60 | **VERIFIED** |
| **10** | Amygdala | Left | Primary AD Target | 306 | 2.45 | **VERIFIED** |
| **20** | Amygdala | Right | Primary AD Target | 366 | 2.93 | **VERIFIED** |
| **3** | Lateral Ventricle | Left | Primary AD Target | 1,145 | 9.16 | **VERIFIED** |
| **14** | Lateral Ventricle | Right | Primary AD Target | 1,019 | 8.15 | **VERIFIED** |
| **4** | Thalamus | Left | Control Structure | 1,321 | 10.57 | **VERIFIED** |
| **15** | Thalamus | Right | Control Structure | 1,283 | 10.26 | **VERIFIED** |
| **5** | Caudate | Left | Control Structure | 487 | 3.90 | **VERIFIED** |
| **16** | Caudate | Right | Control Structure | 511 | 4.09 | **VERIFIED** |
| **6** | Putamen | Left | Control Structure | 799 | 6.39 | **VERIFIED** |
| **17** | Putamen | Right | Control Structure | 796 | 6.37 | **VERIFIED** |

*Note on Voxel Counts*: Because the atlas was applied in standardized MNI space without subject-specific non-linear inverse deformation, the voxel count for every structure is mathematically identical across all 564 participants (`std(nvox) == 0.0`). The `*_nvox` columns do not represent subject-specific morphometric volumes and cannot serve as morphometric features.

### 2.3 Feature Mathematical Formulation & Normalization Audit

1. **Prior Preprocessing Normalization**:
   In `scripts/mri_preprocess_v2.py#L58-60`, raw scans were skull-stripped, robustly field-of-view cropped, bias-corrected via N4, intensity-clipped at $[0.5\%, 99.5\%]$, and z-score normalized across the whole-brain mask:
   $$Z_{\text{vol}} = \frac{X_{\text{clipped}} - \mu_{\text{brain}}}{\sigma_{\text{brain}}}$$
   Thus, the volumes entering Stage 19 were already zero-mean and unit-variance across intracranial voxels.
2. **White Matter Median Subtraction**:
   In `scripts/extract_mri_subcortical_rois.py#L119-122`, bilateral cerebral white matter voxels ($N_{\text{vox}} = 62,732$) were sampled:
   $$\text{WM}_{\text{median}} = \text{median}(Z_{\text{vol}}[\text{Mask}_{\text{WM}}])$$
   For each structure $\text{ROI}$:
   - **Median Contrast**: $\tilde{I}_{\text{med}} = \text{median}(Z_{\text{vol}}[\text{ROI}]) - \text{WM}_{\text{median}}$ (**VERIFIED**)
   - **Mean Contrast**: $\tilde{I}_{\text{mean}} = \text{mean}(Z_{\text{vol}}[\text{ROI}]) - \text{WM}_{\text{median}}$ (**VERIFIED**)
   - **Tails**: $\tilde{I}_{10} = p_{10} - \text{WM}_{\text{median}}, \quad \tilde{I}_{90} = p_{90} - \text{WM}_{\text{median}}$ (**VERIFIED**)
   - **Dispersion (Texture)**: $\sigma_{\text{ROI}} = \text{std}(Z_{\text{vol}}[\text{ROI}])$ (**VERIFIED**, no WM subtraction needed)
   - **Interquartile Range**: $\text{IQR}_{\text{ROI}} = p_{75} - p_{25}$ (**VERIFIED**, no WM subtraction needed)
   - **Ventricle-to-Hippocampus Difference**: $\Delta_{\text{vent-hippo}} = \text{LatVent}_{\text{bilat, med}} - \text{Hippo}_{\text{bilat, med}}$ (**VERIFIED**)
3. **Audit of Scanner Drift Claim**:
   - The Stage 19 report claims this subtraction "eliminates inter-subject scanner gain drift and residual coil sensitivity variations."
   - **ISSUE FOUND**: This claim is physically and scientifically overstated. T1-weighted MRI intensities are arbitrary units; T1 contrast between grey matter, white matter, and CSF depends on sequence repetition time ($T_R$), echo time ($T_E$), flip angle ($\alpha$), and scanner field strength ($1.5\text{ T}$ in ADNI1 vs $3.0\text{ T}$ in ADNI2/3). Scalar subtraction of the white matter median provides a relative contrast anchor within a scan, but does not linearize or eliminate sequence-dependent contrast variations across scanners.
4. **Participant Completeness**:
   All 564 development participants possess valid tensors, with `extraction_status == 'OK'`, `qc_flag == 'CLEAN'`, and zero NaN or infinite values (**VERIFIED**).

---

## 3. MODEL INPUT & FEATURE MAPPING AUDIT (TASK 2)

### 3.1 Resolving the 94-Column vs. 16-Feature Discrepancy

The output file `data/processed/features/mri_subcortical_roi_features.csv` has exactly 94 columns. The report notes that Model B and the MRI-only model evaluate 16 features. The audit traced the exact mapping of all 94 columns:

```
Total Columns in CSV: 94
├── Metadata & QC Columns (5 columns)
│   ├── rid, image_id, wm_median, extraction_status, qc_flag
├── Subcortical ROI Structural Statistics (84 columns)
│   ├── 12 Structures × 7 Statistics (mean, median, p10, p90, std, iqr, nvox) = 84
│   │   ├── Target Structures (6 structures × 7 stats = 42 columns):
│   │   │   ├── hippo_left, hippo_right, amygdala_left, amygdala_right, latvent_left, latvent_right
│   │   └── Control Structures (6 structures × 7 stats = 42 columns):
│   │       ├── thalamus_left, thalamus_right, caudate_left, caudate_right, putamen_left, putamen_right
└── Bilateral & Composite Metrics (5 columns)
    ├── hippo_bilateral_median, hippo_bilateral_mean, amygdala_bilateral_median,
    ├── latvent_bilateral_median, vent_to_hippo_diff
```

### 3.2 Complete Feature Mapping & Model Input Audit

| Column Group | Total Columns | Columns Used in Model B (Primary) | Columns Used in Model C (Extended) | Columns Used in MRI-Only | Reason for Exclusion from Primary Model B |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Metadata / QC** | 5 | 0 | 0 | 0 | Non-predictive identification and quality flags |
| **Target Medians** | 6 | **6** | **6** | **6** | Primary regional tissue contrast markers |
| **Target Stds** | 6 | **6** | **6** | **6** | Primary regional tissue heterogeneity / texture markers |
| **Target Composites** | 5 | **4** | **4** | **4** | Bilateral medians (3) + Vent-to-Hippo diff (1) |
| **Target Means, Tails, IQRs** | 24 | 0 | 0 | 0 | Highly collinear with median and std ($r > 0.95$) |
| **Target Voxel Counts (`nvox`)** | 6 | 0 | 0 | 0 | Zero variance (identical constant across all subjects) |
| **Control Medians** | 6 | 0 | **6** | 0 | Reserved strictly for negative-control Model C |
| **Control Other Stats** | 36 | 0 | 0 | 0 | Redundant / uninformative control statistics |
| **Total Features** | **94** | **16** | **22** | **16** | **100% of 94 columns verified and accounted for** |

### 3.3 The 16 Primary Subcortical MRI Features (`target_roi_cols`)
Traced in `scripts/experiments/run_mri_roi_incremental.py#L83-91`:
1. `hippo_left_median`
2. `hippo_right_median`
3. `hippo_bilateral_median`
4. `hippo_left_std`
5. `hippo_right_std`
6. `amygdala_left_median`
7. `amygdala_right_median`
8. `amygdala_bilateral_median`
9. `amygdala_left_std`
10. `amygdala_right_std`
11. `latvent_left_median`
12. `latvent_right_median`
13. `latvent_bilateral_median`
14. `latvent_left_std`
15. `latvent_right_std`
16. `vent_to_hippo_diff`

### 3.4 Leakage Check on Feature Selection
- **Status: VERIFIED LEAK-FREE**.
- The 16 features were selected *a priori* based on the canonical anatomical vulnerability pattern of Alzheimer's disease (medial temporal lobe atrophy and lateral ventricular expansion).
- No data-driven feature selection (e.g., LASSO, recursive feature elimination, or univariate filtering) was applied to the training or validation folds.
- Outer validation labels were not accessed during feature definition.

---

## 4. COHORT AND SPLIT INTEGRITY AUDIT (TASK 3)

| Audit Verification Item | Expected / Protocol Requirement | Observed in Repository Artifacts | Status | Evidence / Source |
| :--- | :--- | :--- | :---: | :--- |
| **Matched Dev Cohort Size** | Exactly 564 participants | Exactly 564 participants | **VERIFIED** | `s_rids = dev_rids & roi.index`, len = 564 |
| **Outcome Label Distribution** | Binary 24-month progression | 123 Converters (21.81%), 441 Non-converters (78.19%) | **VERIFIED** | `cohort_v2b.csv#label_primary`, sum = 123 |
| **Participant Identity Across Models** | Exact same 564 participants | Identical across Clin-only, Model A, B, C, MRI, GBM | **VERIFIED** | `oof_predictions.csv`, 564 unique RIDs |
| **Split Structure** | 3 repeats $\times$ 5 folds | Exactly 3 repeats (0, 1, 2) and 5 folds (0-4) | **VERIFIED** | `splits_v1.csv`, identical fold partitions |
| **Train / Outer Validation Separation** | Zero participant overlap within folds | 0 participants shared between train and val | **VERIFIED** | Verified across all 15 outer fold splits |
| **OOF Prediction Record Count** | Exactly $564 \times 3 = 1,692$ rows | Exactly 1,692 rows in `oof_predictions.csv` | **VERIFIED** | Groupby check: 564 per repeat, 0 duplicates |
| **Locked Holdout Isolation** | Exactly 0 holdout participants ($N=156$) | Exactly 0 holdout participants present ($0 / 156$) | **VERIFIED** | `len(set(s_rids) & test_rids) == 0` |

---

## 5. INDEPENDENT METRIC REPRODUCTION (TASK 4)

All metrics were independently recalculated directly from `results/mri_roi_incremental/oof_predictions.csv` using `.venv/bin/python` with `scikit-learn 1.5.2`.

### 5.1 Multi-Repeat AUROC, PR-AUC, and Brier Score Reproduction

| Model Configuration | Scope | Reported AUROC | Recalculated AUROC | Discrepancy | Reported PR-AUC | Recalculated PR-AUC | Reported Brier | Recalculated Brier | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clinical Only (10)** | Repeat 0 | 0.7994 | 0.7994 | 0.0000 | 0.5284 | 0.5284 | 0.1425 | 0.1425 | **VERIFIED** |
| | Repeat 1 | 0.8154 | 0.8154 | 0.0000 | 0.5583 | 0.5583 | 0.1408 | 0.1408 | **VERIFIED** |
| | Repeat 2 | 0.8101 | 0.8101 | 0.0000 | 0.5492 | 0.5492 | 0.1410 | 0.1410 | **VERIFIED** |
| | **3-Rep Mean $\pm$ SD** | **0.8083 $\pm$ 0.0066** | **0.8083 $\pm$ 0.0066** | **0.0000** | **0.5453** | **0.5453** | **0.1414** | **0.1414** | **VERIFIED** |
| **Model A: Clin + FS29** | Repeat 0 | 0.8203 | 0.8203 | 0.0000 | 0.6083 | 0.6083 | 0.1235 | 0.1235 | **VERIFIED** |
| | Repeat 1 | 0.8697 | 0.8697 | 0.0000 | 0.6687 | 0.6687 | 0.1133 | 0.1133 | **VERIFIED** |
| | Repeat 2 | 0.8691 | 0.8691 | 0.0000 | 0.6670 | 0.6670 | 0.1138 | 0.1138 | **VERIFIED** |
| | **3-Rep Mean $\pm$ SD** | **0.8530 $\pm$ 0.0231** | **0.8530 $\pm$ 0.0231** | **0.0000** | **0.6480** | **0.6480** | **0.1169** | **0.1169** | **VERIFIED** |
| **Model B: Clin+FS+ROI** | Repeat 0 | 0.8467 | 0.8467 | 0.0000 | 0.6391 | 0.6391 | 0.1176 | 0.1176 | **VERIFIED** |
| | Repeat 1 | 0.8726 | 0.8726 | 0.0000 | 0.6865 | 0.6865 | 0.1118 | 0.1118 | **VERIFIED** |
| | Repeat 2 | 0.8712 | 0.8712 | 0.0000 | 0.6798 | 0.6798 | 0.1123 | 0.1123 | **VERIFIED** |
| | **3-Rep Mean $\pm$ SD** | **0.8635 $\pm$ 0.0119** | **0.8635 $\pm$ 0.0119** | **0.0000** | **0.6685** | **0.6685** | **0.1139** | **0.1139** | **VERIFIED** |
| **Model C: +Control ROI** | Repeat 0 | 0.8260 | 0.8260 | 0.0000 | 0.6133 | 0.6133 | 0.1237 | 0.1237 | **VERIFIED** |
| | Repeat 1 | 0.8698 | 0.8698 | 0.0000 | 0.6698 | 0.6698 | 0.1136 | 0.1136 | **VERIFIED** |
| | Repeat 2 | 0.8707 | 0.8707 | 0.0000 | 0.6657 | 0.6657 | 0.1139 | 0.1139 | **VERIFIED** |
| | **3-Rep Mean $\pm$ SD** | **0.8555 $\pm$ 0.0209** | **0.8555 $\pm$ 0.0209** | **0.0000** | **0.6496** | **0.6496** | **0.1171** | **0.1171** | **VERIFIED** |
| **MRI ROI Only (16)** | Repeat 0 | 0.6861 | 0.6861 | 0.0000 | 0.3842 | 0.3842 | 0.1585 | 0.1585 | **VERIFIED** |
| | Repeat 1 | 0.7203 | 0.7203 | 0.0000 | 0.4150 | 0.4150 | 0.1531 | 0.1531 | **VERIFIED** |
| | Repeat 2 | 0.7242 | 0.7242 | 0.0000 | 0.4162 | 0.4162 | 0.1524 | 0.1524 | **VERIFIED** |
| | **3-Rep Mean $\pm$ SD** | **0.7102 $\pm$ 0.0171** | **0.7102 $\pm$ 0.0171** | **0.0000** | **0.4051** | **0.4051** | **0.1547** | **0.1547** | **VERIFIED** |
| **Model B Nonlinear GBM**| Repeat 0 | 0.8507 | 0.8507 | 0.0000 | 0.6436 | 0.6436 | 0.1177 | 0.1177 | **VERIFIED** |
| | Repeat 1 | 0.8491 | 0.8491 | 0.0000 | 0.6409 | 0.6409 | 0.1189 | 0.1189 | **VERIFIED** |
| | Repeat 2 | 0.8574 | 0.8574 | 0.0000 | 0.6559 | 0.6559 | 0.1165 | 0.1165 | **VERIFIED** |
| | **3-Rep Mean $\pm$ SD** | **0.8524 $\pm$ 0.0036** | **0.8524 $\pm$ 0.0036** | **0.0000** | **0.6468** | **0.6468** | **0.1177** | **0.1177** | **VERIFIED** |

### 5.2 Pooled Repeat-Averaged Participant-Level Metrics ($N = 564$)
Each participant's out-of-fold predicted probabilities across the 3 repeats were averaged to obtain one stable prediction vector ($N = 564$):

| Model Configuration | Pooled AUROC | Pooled PR-AUC | Pooled Brier Score | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Clinical Only (10)** | 0.8210 | 0.5488 | 0.1342 | **VERIFIED** |
| **Model A: Clinical + FS29** | 0.8643 | 0.6598 | 0.1132 | **VERIFIED** |
| **Model B: Clinical + FS29 + MRI ROI** | 0.8688 | 0.6756 | 0.1116 | **VERIFIED** |
| **Model C: Clinical + FS29 + ROI + Ctrl** | 0.8640 | 0.6671 | 0.1139 | **VERIFIED** |
| **MRI ROI Only (16)** | 0.7183 | 0.4079 | 0.1534 | **VERIFIED** |
| **Model B Nonlinear GBM** | 0.8616 | 0.6655 | 0.1139 | **VERIFIED** |

*Important Aggregation Note*: The 3-repeat mean AUROC for Model B is **0.8635**, whereas the pooled repeat-averaged AUROC is **0.8688**. These two figures represent different aggregation methods and must not be used interchangeably. The paired bootstrap analysis in Section 5.3 is based strictly on the pooled repeat-averaged predictions ($N = 564$).

### 5.3 Paired Bootstrap Reproduction ($N = 564, B = 2,000$, Seed 42)

| Contrast Description | Metric | Reported Point Delta | Recalculated Point Delta | Reported 95% CI | Recalculated 95% CI | Reported $p$-Value | Recalculated $p$-Value | Verification Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Primary: Model B minus Model A** | AUROC | **+0.0044** | **+0.0044** | [**-0.0029**, **+0.0122**] | [**-0.0029**, **+0.0122**] | **0.2440** | **0.2440** | **VERIFIED** |
| | PR-AUC | +0.0073 | +0.0073 | [-0.0271, +0.0471] | [-0.0271, +0.0471] | 0.7140 | 0.7140 | **VERIFIED** |
| | Brier | -0.0038 | -0.0038 | [-0.0081, +0.0004] | [-0.0081, +0.0004] | 0.0790 | 0.0790 | **VERIFIED** |
| **Control: Model C minus Model A** | AUROC | **-0.0003** | **-0.0003** | [**-0.0093**, **+0.0091**] | [**-0.0093**, **+0.0091**] | **0.8770** | **0.8770** | **VERIFIED** |
| **Validation: Model A minus Clin** | AUROC | **+0.0433** | **+0.0433** | [**+0.0135**, **+0.0747**] | [**+0.0135**, **+0.0747**] | **0.0020** | **0.0020** | **VERIFIED** |
| **Nonlinear: GBM minus Model A** | AUROC | **-0.0028** | **-0.0028** | [**-0.0212**, **+0.0158**] | [**-0.0212**, **+0.0158**] | **0.7560** | **0.7560** | **VERIFIED** |

---

## 6. BOOTSTRAP METHODOLOGY AUDIT (TASK 5)

The bootstrap implementation in `scripts/experiments/run_mri_roi_incremental.py#L394-440` was evaluated line-by-line:

| Evaluation Aspect | Protocol Verification Finding | Methodological Classification | Impact on Scientific Validity |
| :--- | :--- | :---: | :--- |
| **Resampling Unit** | Participant-level ($N = 564$) | **VERIFIED** | Correct; prevents pseudo-replication from multi-repeat observations. |
| **Pairing Preservation** | In each iteration $b$, identical index vector `idx` is evaluated for both models: $\Delta^{(b)} = \text{AUC}_{\text{test}}[\text{idx}] - \text{AUC}_{\text{ref}}[\text{idx}]$ | **VERIFIED** | Correct; properly models the covariance between competing model predictions. |
| **Outcome Stratification** | **NOT STRATIFIED**. Uniform sampling with replacement (`rng.integers(0, n, n)`) with acceptance check `len(np.unique(y_true[idx])) > 1` | **ISSUE FOUND** | While unstratified resampling is acceptable, stratified bootstrap resampling is standard in class-imbalanced biomedical settings ($21.8\%$ positive). |
| **Sample Size & Seed** | $B = 2,000$ iterations, `np.random.default_rng(42)` | **VERIFIED** | Sufficient sample size; perfectly reproducible. |
| **Confidence Interval Method** | Empirical Percentile Method ($2.5^{\text{th}}$ and $97.5^{\text{th}}$ percentiles of bootstrap deltas) | **VERIFIED** | Standard non-parametric confidence interval formulation. |
| **$p$-Value Formulation** | Empirical tail proportion: $p = 2 \times \min(P(\Delta \le 0), P(\Delta \ge 0))$ | **ISSUE FOUND** | This is an **empirical bootstrap-tail proportion**, not a conventional parametric hypothesis-test $p$-value. It represents the proportion of bootstrap resamples falling on the opposite side of zero doubled for two-sidedness. The paper must explicitly label it as an empirical bootstrap-tail $p$-value. |

---

## 7. INNER VALIDATION AND LEAKAGE AUDIT (TASK 6)

The nested inner validation routine (`run_mri_roi_incremental.py#L229-272`) was audited:

```
Outer Fold Training Set (N ≈ 451)
├── Nested Inner Split: 85% Inner Train (N ≈ 383) / 15% Inner Val (N ≈ 68)
│   ├── Step A: prep_fold_matrices(itr, iva) -> Fit scaler on itr, evaluate C ∈ C_GRID on iva -> Pick best_c
│   ├── Step B: Fit model(best_c) on itr -> Predict on iva -> Youden J = Sens + Spec - 1 -> Pick t_youden
├── Step C: prep_fold_matrices(trn, val) -> Fit scaler on trn -> Fit model(best_c) on trn
└── Step D: Predict on outer val -> Evaluate AUROC & threshold metrics(t_youden)
```

### 7.1 Verification of Leak-Free Guarantees
1. **Outer Validation Purity (VERIFIED)**: The outer validation fold ($N \approx 113$) was held out entirely during hyperparameter $C$ tuning, threshold selection, and scaling estimation.
2. **Standardization & Imputation Leak-Free (VERIFIED)**: Feature medians and standard deviations were computed strictly on the training partition and applied forward.
3. **Threshold-Independent Metrics (VERIFIED)**: AUROC and PR-AUC depend solely on the relative ranking of continuous probabilities and are completely unaffected by threshold choice.

### 7.2 Methodological Critique of Threshold-Dependent Metrics
- **Dual-Use of Inner Validation (ISSUE FOUND)**: The same 15% inner validation split ($N \approx 68$) was used first to tune $C$, and subsequently to choose Youden's $J$ threshold. Because $C$ was chosen to optimize discrimination on this sample, evaluating the threshold on the same sample introduces slight optimism.
- **Refitting Probability Calibration Shift (ISSUE FOUND)**: In Step C, the model was refit on the full outer training set ($N \approx 451$), which is $18\%$ larger than the inner training set ($N \approx 383$). When refit on more data, logistic regression coefficients can shift, altering probability calibration. Applying the threshold $t_{\text{youden}}$ derived from the inner model to the outer refit model introduces operating threshold mismatch.
- **Status of Operating Metrics**: The fold-level threshold-dependent metrics (Accuracy $77.98\%$, Sensitivity $73.67\%$, Specificity $79.28\%$) are **unbiased with respect to the outer validation labels**, but sub-optimal due to the calibration shift. In the research paper, primary conclusions must rest on AUROC and PR-AUC.

---

## 8. ADNI PHASE ANALYSIS AUDIT (TASK 7)

Traced from `results/mri_roi_incremental/phase_breakdown.csv` and cross-referenced with `data/processed/cohort_v2b.csv`:

| ADNI Phase | Total Cohort ($N$) | Converters ($N_{\text{pos}}$) | Non-Converters ($N_{\text{neg}}$) | Conversion Prevalence | Model A AUROC | Model B AUROC | Delta AUROC (B minus A) | Audit Classification & Stability Assessment |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **ADNI1** | 188 | 72 | 116 | 38.3% | 0.7981 | 0.7968 | **-0.0013** | **STABLE**: Adequate sample and event counts. Shows zero/negative increment. |
| **ADNI2** | 204 | 38 | 166 | 18.6% | 0.8519 | 0.8581 | **+0.0062** | **STABLE**: Adequate sample size. Shows negligible increment ($< 0.01$). |
| **ADNI3** | 93 | 10 | 83 | 10.8% | 0.8627 | 0.8904 | **+0.0277** | **HIGHLY UNSTABLE**: Only 10 positive events. Severe sampling error. |
| **ADNIGO** | 79 | 3 | 76 | 3.8% | 0.9430 | 0.9474 | **+0.0044** | **HIGHLY UNSTABLE**: Only 3 positive events. AUROC is statistically uninterpretable. |

### Critique of Phase Interpretation:
- In ADNI3 ($N_{\text{pos}} = 10$), a single participant misclassification alters the AUROC by approximately $0.10$.
- In ADNIGO ($N_{\text{pos}} = 3$), the sample is so underpowered that AUROC estimates carry wide error bounds.
- **MANDATORY CORRECTION**: The nominal $+0.0277$ delta in ADNI3 must **NOT** be reported as evidence of an emerging MRI benefit or modern scanner superiority. In the two adequately powered cohorts (ADNI1 and ADNI2, totaling $N = 392, 110$ converters), the MRI increment is $-0.0013$ and $+0.0062$, confirming an absence of incremental predictive value.

---

## 9. SCIENTIFIC INTERPRETATION AUDIT (TASK 8)

The audit systematically evaluated the six core scientific assertions relevant to the UROP research paper:

```
Scientific Assertion Status Matrix:
[1] Clin + FS29 > Clin Only               ──> [SUPPORTED]        (Δ = +0.0433, p = 0.0020)
[2] Subcortical MRI Small Positive Delta   ──> [SUPPORTED]        (Δ = +0.0044, point-estimate)
[3] MRI Increment Not Statistically Sig    ──> [SUPPORTED]        (95% CI [-0.0029, +0.0122], p = 0.2440)
[4] MRI Features Share Info with FS        ──> [SUPPORTED]        (|r| ≈ 0.30 - 0.36, p < 0.001)
[5] MRI Info Completely Captured by FS     ──> [NOT SUPPORTED]    (Logically and statistically flawed claim)
[6] Baseline MRI Has No Biological Value   ──> [NOT SUPPORTED]    (Scientifically invalid claim)
```

### Detailed Evaluation of Each Statement:

1. **"Clinical + FS29 outperforms Clinical only on the matched cohort."**
   - **Verdict: SUPPORTED**.
   - **Evidence**: Model A (Clinical + FS29) achieves pooled AUROC of $0.8643$ vs. Clinical Only $0.8210$. Paired participant-level bootstrap yields $\Delta = +0.0433$ [95% CI: $+0.0135, +0.0747$], $p = 0.0020$. FreeSurfer regional morphometry provides a statistically robust, reproducible increment.
2. **"Adding subcortical MRI ROI features produces a small positive AUROC point-estimate difference."**
   - **Verdict: SUPPORTED**.
   - **Evidence**: Model B achieves pooled AUROC of $0.8688$ vs. Model A's $0.8643$ ($\Delta = +0.0044$). Across the 15 outer folds, the mean delta is $+0.0077 \pm 0.0178$. The empirical point estimate is positive, but very small.
3. **"The MRI increment is not statistically significant."**
   - **Verdict: SUPPORTED**.
   - **Evidence**: The primary paired bootstrap 95% CI spans across zero ($[-0.0029, +0.0122]$), with empirical tail $p = 0.2440$. The fold-level paired $t$-test across 15 folds yields $t = 1.6745, p = 0.1162$. Both tests fail to reject the null hypothesis at $\alpha = 0.05$.
4. **"The MRI features share some information with FreeSurfer volumetry."**
   - **Verdict: SUPPORTED**.
   - **Evidence**: Verified Pearson correlations between MNI subcortical ROI intensity features and native FreeSurfer volumetric measures demonstrate moderate, statistically significant concordance ($p < 0.001$):
     - Left Hippocampus contrast vs. FS volume (`ST29SV`): $r = +0.3002$
     - Right Hippocampus contrast vs. FS volume (`ST88SV`): $r = +0.3132$
     - Left Lateral Ventricle contrast vs. FS volume (`ST37SV`): $r = -0.3622$
     - Right Lateral Ventricle contrast vs. FS volume (`ST96SV`): $r = -0.3661$
5. **"MRI predictive information is completely captured by FreeSurfer."**
   - **Verdict: NOT SUPPORTED (OVERSTATED / SCIENTIFICALLY FLAWED)**.
   - **Critique**: Failure to reject the null hypothesis of no incremental value ($p = 0.2440$) does **not** prove mathematical equivalence or complete redundancy ("absence of evidence is not evidence of absence"). Formal claims of equivalence require a Two One-Sided Tests (TOST) equivalence framework with pre-specified equivalence margins ($\pm \delta$). Furthermore, Stage 19 extracted linear standard-space intensity statistics; it did not evaluate non-linear morphology, cortical thickness, or deep representations. Claiming that FreeSurfer "completely captures" all MRI predictive information is an overstatement.
6. **"Baseline MRI has no additional biological value."**
   - **Verdict: NOT SUPPORTED (CATEGORICALLY FALSE / SCIENTIFICALLY INVALID)**.
   - **Critique**: Statistical non-significance in a specific 24-month linear progression model cannot be equated with a lack of biological value. Structural MRI is biologically informative: it reflects true macroscopic atrophy, exhibits validated correlations with FreeSurfer markers, and is indispensable in clinical practice for differential diagnosis (e.g., vascular pathology, subdural hematoma, hydrocephalus). The finding indicates that in the presence of sensitive clinical cognitive tests and subject-space FreeSurfer morphometry, subcortical linear intensity features provide no incremental statistical signal over a 24-month prediction window.

---

## 10. LIST OF UNRESOLVED ISSUES AND RECOMMENDED CORRECTIONS

| Issue ID | Identified Flaw or Limitation | Classification | Exact Source in Repository | Recommended Manuscript Correction |
| :---: | :--- | :---: | :--- | :--- |
| **ISS-01** | Linear affine registration (12 DOF) lacks non-linear deformable alignment | **LIMITATION** | `mri_preprocess_v2.py#L26-35` | Acknowledge in Discussion: subcortical ROIs in $2\text{ mm}$ MNI space experience partial volume averaging at boundaries due to unmodeled individual anatomical deformation. |
| **ISS-02** | Overstated claim that white matter subtraction "eliminates scanner effects" | **ISSUE FOUND** | `reports/mri_subcortical_roi_extraction_report.md#L45` | Revise text to state that white matter median referencing provides an *intra-subject relative baseline shift*, but does not eliminate scanner/sequence-dependent contrast variations across ADNI phases. |
| **ISS-03** | Lack of a complete feature mapping table for the 94 extracted columns | **ISSUE FOUND** | `reports/mri_subcortical_roi_extraction_report.md` | Include the feature mapping table from Section 3.2 of this audit, detailing why 78 columns were excluded from Model B. |
| **ISS-04** | Dual-use of inner validation split for tuning $C$ and Youden $J$ threshold | **METHODOLOGICAL ISSUE** | `run_mri_roi_incremental.py#L241-256` | Clarify in Methods that operating threshold metrics are secondary and may carry minor calibration shift; base primary conclusions strictly on AUROC and PR-AUC. |
| **ISS-05** | Fragile sample size in ADNI3 ($N_{\text{pos}}=10$) and ADNIGO ($N_{\text{pos}}=3$) | **INTERPRETATION ISSUE** | `phase_breakdown.csv` | Do not claim that ADNI3 shows an established MRI benefit ($+0.0277$). Explicitly state that ADNI3 and ADNIGO are underpowered and statistically unstable. |
| **ISS-06** | Claims of "complete capture" and "no biological value" | **SCIENTIFIC OVERSTATEMENT** | `STAGE19_MRI_ROI_INCREMENTAL_REPORT.md#L185-195` | Replace absolute claims of "complete redundancy" and "no biological value" with nuanced language: failure to demonstrate statistically significant incremental predictive value ($p = 0.2440$) beyond Clinical + FS29 in a 24-month progression window. |

---

## 11. FINAL CONCLUSION AND DISPOSITION

### Recommendation: **ACCEPT WITH CORRECTIONS**

Stage 19 is fully verified in its numerical execution, code integrity, and data isolation. The primary research question has received an empirical answer:

$$\Delta_{\text{AUROC}} = +0.0044 \quad [95\% \text{ CI: } -0.0029, +0.0122], \quad p = 0.2440$$

Baseline 3D subcortical MRI ROI intensity and heterogeneity features do not provide a statistically significant incremental benefit beyond clinical testing and FreeSurfer summary morphometry for 24-month MCI-to-AD progression.

The experiment is approved for inclusion in the B.Tech UROP research thesis, provided the specific textual and scientific corrections detailed in Section 10 are incorporated into the final manuscript.
