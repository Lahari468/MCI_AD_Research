# STAGE 19: TARGETED MRI ROI CONTRIBUTION EXPERIMENT REPORT

**Title**: Evaluating the Incremental Predictive Value of Anatomically Informed Baseline 3D MRI Subcortical ROI Features Beyond Clinical and FreeSurfer Morphometry for 24-Month MCI-to-AD Progression  
**Date**: November 23, 2026  
**Repository**: `.`  
**Output Directory**: `results/mri_roi_incremental/`  
**Target Outcome**: 24-Month MCI-to-AD Progression (`label_primary`, 730-day follow-up horizon)  
**Cohort Scope**: Strictly Development Participants with verified complete baseline 3D T1 MRI ($N = 564$, 123 converters, 441 non-converters). Locked holdout ($N = 156$) strictly excluded.

---

## 1. RESEARCH QUESTION & HYPOTHESIS

* **Primary Research Question**: Does an anatomically informed subcortical MRI representation (regional intensity contrast and heterogeneity extracted from standardized MNI152 volumes) provide statistically significant incremental predictive value beyond validated baseline clinical assessments and FreeSurfer summary morphometry for 24-month MCI progression?
* **Ablation Hypothesis**: If subcortical T1 intensity features contain predictive biological signal orthogonal to FreeSurfer morphometry (e.g. tissue contrast, signal alteration, or subtle ventricular/periventricular boundary shifts), Model B (Clinical 10 + FreeSurfer FS29 + MRI ROI) will statistically outperform Model A (Clinical 10 + FreeSurfer FS29) under a matched, leakage-free 3-repeat 5-fold cross-validation protocol.

---

## 2. COHORT, ATLAS & SPATIAL FEASIBILITY AUDIT

### 2.1 Matched Participant Cohort ($N = 564$)
* Starting screened cohort: 1,199 participants.
* Primary eligible development cohort: 622 participants (144 converters, 478 non-converters).
* **Usable Baseline 3D T1 MRI on Disk**: Exactly **564 participants** (123 converters, 441 non-converters, 21.81% prevalence).
* Both Model A and Model B were evaluated on the **exact same 564 participants** across all 15 outer folds.

### 2.2 Atlas and Coordinate Space Verification
* **Atlas Used**: Harvard-Oxford Subcortical Structural Atlas (probabilistic, maximum probability thresholded at 25%).
* **Coordinate Space**: MNI152 Linear Standard Space, 2 mm isotropic voxel resolution ($91 \times 109 \times 91$ grid).
* **Affine Alignment**: The preprocessed MRI volumes in `data/interim/mri_mni2mm_v2/<RID>.npy` were linearly registered via FLIRT (12 DOF) to `MNI152_T1_2mm.nii.gz` with N4 bias-field correction and brain-mask z-scoring. The spatial orientation, dimensions, and affine matrix match the Harvard-Oxford atlas bit-for-bit.
* **Cortical Limitation (Verified & Documented)**:
  * Entorhinal, parahippocampal, fusiform, temporal, and precuneus labels are **NOT AVAILABLE** in the local subcortical atlas files.
  * In accordance with scientific integrity guidelines, no artificial cortical atlases or synthetic transformations were invented.
  * Cortical morphometry is already natively represented in subject space via the 29 FreeSurfer features (`freesurfer29_raw_v1.csv`).

### 2.3 Subcortical Structure Mapping

| Anatomical Structure | Hemisphere | Target Role | Voxel Count (2 mm) | Volume ($\text{cm}^3$) |
| :--- | :---: | :---: | :---: | :---: |
| **Cerebral White Matter** | Bilateral (L: 1, R: 12) | Intensity Calibration Reference | 62,732 | 501.86 |
| **Hippocampus** | Left (9), Right (19) | Primary AD Target | 691 / 700 | 5.53 / 5.60 |
| **Amygdala** | Left (10), Right (20) | Primary AD Target | 306 / 366 | 2.45 / 2.93 |
| **Lateral Ventricles** | Left (3), Right (14) | Primary AD Target | 1,145 / 1,019 | 9.16 / 8.15 |
| **Thalamus** | Left (4), Right (15) | Control Structure | 1,321 / 1,283 | 10.57 / 10.26 |
| **Caudate** | Left (5), Right (16) | Control Structure | 487 / 511 | 3.90 / 4.09 |
| **Putamen** | Left (6), Right (17) | Control Structure | 799 / 796 | 6.39 / 6.37 |

---

## 3. FEATURE EXTRACTION & PREPROCESSING METHODOLOGY

### 3.1 Feature Formulation
In the preprocessed volumes, voxel intensities were globally brain-masked and z-scored ($X = (x - \mu_{\text{brain}}) / \sigma_{\text{brain}}$). To eliminate inter-scanner gain drift and coil sensitivity variations, each regional statistic was referenced to the median intensity of bilateral cerebral white matter ($X_{\text{WM}}$, 62,732 voxels):
$$\text{WM}_{\text{median}} = \text{median}(X_{\text{WM}})$$

For each region of interest $\text{ROI}$:
1. **Median Contrast**: $\tilde{I}_{50} = \text{median}(X_{\text{ROI}}) - \text{WM}_{\text{median}}$
2. **Mean Contrast**: $\tilde{I}_{\text{mean}} = \text{mean}(X_{\text{ROI}}) - \text{WM}_{\text{median}}$
3. **Internal Dispersion (Texture/Heterogeneity)**: $\sigma_{\text{ROI}} = \text{std}(X_{\text{ROI}})$
4. **Extreme Tails**: $\tilde{I}_{10} = p_{10}(X_{\text{ROI}}) - \text{WM}_{\text{median}}, \quad \tilde{I}_{90} = p_{90}(X_{\text{ROI}}) - \text{WM}_{\text{median}}$
5. **Ventricle-to-Hippocampus Difference**: $\Delta_{\text{vent-hippo}} = \text{Ventricle}_{\text{bilateral, median}} - \text{Hippo}_{\text{bilateral, median}}$

*Total Extracted Primary ROI Features*: 16 compact, anatomically motivated features (`target_roi_cols`).

### 3.2 Feature Sets Evaluated
1. **Clinical Only**: 10 primary baseline features (age, sex, education, APOE4, MMSE, CDR-SB, CDR-Global, FAQ, ADAS11, ADAS13).
2. **Model A (Primary Baseline)**: Clinical 10 + FreeSurfer FS29 (39 features total, with volumes normalized by intracranial volume `ST10CV * 1000.0`).
3. **Model B (Primary Targeted Contrast)**: Clinical 10 + FreeSurfer FS29 + 16 Primary Subcortical MRI ROI features (55 features total).
4. **Model C (Extended Control Contrast)**: Clinical 10 + FreeSurfer FS29 + 16 Primary Subcortical ROI features + 6 Anatomical Control ROI features (Thalamus, Caudate, Putamen medians; 61 features total).
5. **MRI ROI Only**: 16 Subcortical MRI ROI features alone.
6. **Model B Nonlinear GBM**: HistGradientBoostingClassifier evaluated on Model B features.

---

## 4. EXPERIMENTAL CONTROLS & LEAKAGE PREVENTION

1. **Identical Folds Across Repeats**: Driven strictly by `data/processed/splits/splits_v1.csv` for the 564 participants across Repeat 0, Repeat 1, and Repeat 2.
2. **Nested Inner Validation**: In each of the 15 outer folds ($\approx 451$ training cases), an inner 15% stratified split was used to select the optimal L2 regularization parameter $C \in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]$ via AUROC.
3. **Inner-Fold Feature Scaling & Imputation**: Median imputation and z-score standardization were fitted strictly on the inner training folds and applied forward to validation folds.
4. **Threshold Selection**: Classification thresholds were frozen strictly on inner validation predictions using Youden's $J = \text{Sens} + \text{Spec} - 1$, preventing outer validation leakage.
5. **Locked Holdout Immunity**: Zero locked holdout participants or labels were opened, referenced, or evaluated.

---

## 5. EMPIRICAL RESULTS

### 5.1 Multi-Repeat Summary (3 Repeats $\times$ 5 Folds, $N = 564$)

| Model Configuration | Repeat 0 AUROC | Repeat 1 AUROC | Repeat 2 AUROC | 3-Repeat Mean $\pm$ SD | Mean PR-AUC | Mean Brier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clinical Only (10 feats)** | 0.7994 | 0.8154 | 0.8101 | **0.8083 $\pm$ 0.0066** | 0.5453 | 0.1414 |
| **Model A: Clinical + FS29** | 0.8203 | 0.8697 | 0.8691 | **0.8530 $\pm$ 0.0231** | 0.6480 | 0.1169 |
| **Model B: Clin + FS29 + MRI ROI** | 0.8467 | 0.8726 | 0.8712 | **0.8635 $\pm$ 0.0119** | 0.6685 | 0.1139 |
| **Model C: Clin + FS29 + ROI + Ctrl** | 0.8260 | 0.8698 | 0.8707 | **0.8555 $\pm$ 0.0209** | 0.6496 | 0.1171 |
| **MRI ROI Only (16 feats)** | 0.6861 | 0.7203 | 0.7242 | **0.7102 $\pm$ 0.0171** | 0.4051 | 0.1547 |
| **Model B Nonlinear GBM** | 0.8507 | 0.8491 | 0.8574 | **0.8524 $\pm$ 0.0036** | 0.6468 | 0.1177 |

---

### 5.2 Primary Paired Participant-Level Bootstrap Evaluation ($N = 564, B = 2,000$)

Participant-level out-of-fold predicted probabilities were averaged across the 3 repeats before calculating paired bootstrap differences:

| Contrast Description | Test Model AUROC | Reference Model AUROC | Paired Delta AUROC | 95% Bootstrap CI | Two-Sided $p$-Value | Statistical Significance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Primary: Model B (Clin+FS+ROI) minus Model A (Clin+FS)** | **0.8688** | **0.8643** | **+0.0044** | [**-0.0029**, **+0.0122**] | **p = 0.2440** | **NOT SIGNIFICANT ($p > 0.05$)** |
| **Control: Model C (+Controls) minus Model A (Clin+FS)** | **0.8640** | **0.8643** | **-0.0003** | [**-0.0093**, **+0.0091**] | **p = 0.8770** | **NOT SIGNIFICANT ($p > 0.05$)** |
| **Validation: Model A (Clin+FS) minus Clinical Only** | **0.8643** | **0.8210** | **+0.0433** | [**+0.0135**, **+0.0747**] | **p = 0.0020** | **STATISTICALLY SIGNIFICANT ($p < 0.01$)** |
| **Nonlinear: GBM (Clin+FS+ROI) minus Model A (Clin+FS)** | **0.8616** | **0.8643** | **-0.0028** | [**-0.0212**, **+0.0158**] | **p = 0.7560** | **NOT SIGNIFICANT ($p > 0.05$)** |

* Additional Primary Metric Contrasts (Model B minus Model A):
  * **Delta PR-AUC**: **+0.0073** [95% CI: $-0.0271, +0.0471$], $p = 0.7140$ (Not Significant).
  * **Delta Brier Score**: **-0.0038** [95% CI: $-0.0081, +0.0004$], $p = 0.0790$ (Not Significant).

---

### 5.3 Fold-Level AUROCs Across All 15 Folds

| Repeat | Outer Fold | Model A: Clin + FS29 | Model B: Clin + FS29 + MRI ROI | Fold Delta (Model B minus Model A) |
| :---: | :---: | :---: | :---: | :---: |
| **0** | Fold 0 | 0.7784 | 0.8339 | +0.0555 |
| **0** | Fold 1 | 0.8121 | 0.8465 | +0.0344 |
| **0** | Fold 2 | 0.9253 | 0.9279 | +0.0026 |
| **0** | Fold 3 | 0.8413 | 0.8331 | -0.0082 |
| **0** | Fold 4 | 0.8352 | 0.8368 | +0.0015 |
| **1** | Fold 0 | 0.9014 | 0.9121 | +0.0107 |
| **1** | Fold 1 | 0.8524 | 0.8493 | -0.0031 |
| **1** | Fold 2 | 0.8876 | 0.8876 | 0.0000 |
| **1** | Fold 3 | 0.8017 | 0.8123 | +0.0105 |
| **1** | Fold 4 | 0.9253 | 0.9155 | -0.0098 |
| **2** | Fold 0 | 0.8539 | 0.8635 | +0.0096 |
| **2** | Fold 1 | 0.8708 | 0.8819 | +0.0111 |
| **2** | Fold 2 | 0.8769 | 0.8842 | +0.0073 |
| **2** | Fold 3 | 0.8881 | 0.8974 | +0.0093 |
| **2** | Fold 4 | 0.9038 | 0.8874 | -0.0164 |
| **Summary** | **15 Folds** | **0.8636 $\pm$ 0.0442** | **0.8713 $\pm$ 0.0348** | **+0.0077 $\pm$ 0.0178** |

* **Paired Fold-Level $t$-Test ($K = 15$ folds)**:
  $$t = 1.6745, \quad df = 14, \quad \mathbf{p = 0.1162 \quad (\text{NOT SIGNIFICANT})}$$

---

### 5.4 Threshold-Dependent Operating Metrics (Mean $\pm$ SD across 15 Folds)

Thresholds chosen on inner-validation folds via Youden's $J$:

| Metric | Clinical Only | Model A: Clin + FS29 | Model B: Clin + FS29 + MRI ROI | Delta (B minus A) |
| :--- | :---: | :---: | :---: | :---: |
| **Optimal Threshold ($t$)** | $0.2367 \pm 0.0462$ | $0.2093 \pm 0.0556$ | $0.2580 \pm 0.1160$ | $+0.0487$ |
| **Accuracy** | $0.7609 \pm 0.0667$ | $0.7526 \pm 0.0669$ | $0.7798 \pm 0.0698$ | $+0.0272$ |
| **Balanced Accuracy** | $0.7345 \pm 0.0552$ | $0.7608 \pm 0.0420$ | $0.7647 \pm 0.0541$ | $+0.0039$ |
| **Sensitivity** | $0.6855 \pm 0.1357$ | $0.7750 \pm 0.0865$ | $0.7367 \pm 0.1491$ | $-0.0383$ |
| **Specificity** | $0.7836 \pm 0.1033$ | $0.7467 \pm 0.0964$ | $0.7928 \pm 0.1129$ | $+0.0461$ |
| **Precision (PPV)** | $0.4923 \pm 0.1039$ | $0.4770 \pm 0.0897$ | $0.5378 \pm 0.1437$ | $+0.0608$ |
| **F1-Score** | $0.5571 \pm 0.0721$ | $0.5826 \pm 0.0642$ | $0.5958 \pm 0.0782$ | $+0.0132$ |

---

## 6. SECONDARY INVESTIGATIONS

### 6.1 Biological Correlation with FreeSurfer Volumetry
Pearson correlation between subcortical MRI intensity features and FreeSurfer regional volumes:
* Left Hippocampus Median Contrast vs. FS Left Hippocampal Volume (`ST29SV`): $r = \mathbf{+0.3002}$ ($p < 0.001$).
* Right Hippocampus Median Contrast vs. FS Right Hippocampal Volume (`ST88SV`): $r = \mathbf{+0.3132}$ ($p < 0.001$).
* Left Lateral Ventricle Median Contrast vs. FS Left Ventricular Volume (`ST37SV`): $r = \mathbf{-0.3622}$ ($p < 0.001$).
* Right Lateral Ventricle Median Contrast vs. FS Right Ventricular Volume (`ST96SV`): $r = \mathbf{-0.3661}$ ($p < 0.001$).

*Interpretation*: Regional T1 intensity in standardized MNI space captures expected tissue contrast (hippocampal tissue vs. surrounding hypointense CSF), correlating moderately ($|r| \approx 0.30 - 0.36$) with native-space volumetric atrophy. However, because FreeSurfer morphometry directly quantifies volume in native patient coordinates without spatial resampling blurring, the intensity features from MNI tensors do not add orthogonal biological signal.

### 6.2 ADNI Phase Stratification

| ADNI Phase | Sample Size ($N$) | Converters | Non-Converters | Model A AUROC | Model B AUROC | Delta AUROC (B minus A) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ADNI1** | 188 | 72 | 116 | 0.7981 | 0.7968 | **-0.0013** |
| **ADNI2** | 204 | 38 | 166 | 0.8519 | 0.8581 | **+0.0062** |
| **ADNI3** | 93 | 10 | 83 | 0.8627 | 0.8904 | **+0.0277** |
| **ADNIGO** | 79 | 3 | 76 | 0.9430 | 0.9474 | **+0.0044** |

*Interpretation*: Performance gains are absent in ADNI1 ($-0.0013$), where 1.5T scanner heterogeneity is highest. A modest nominal gain in ADNI3 ($+0.0277$) reflects modern 3.0T resolution but carries high sampling error due to low event counts ($N_{\text{pos}} = 10$).

---

## 7. RESOLUTION OF EXECUTION DEVICE DISCREPANCY

* **Finding**: `torch.backends.mps.is_available()` returns `False` inside the macOS sandbox (due to restricted IOKit/Metal device access), falling back cleanly to CPU (`DEV = 'cpu'`). Outside the sandbox, MPS evaluates to `True`.
* **Verification**: All training in Stage 12 and Stage 19 ran deterministically on CPU across physical cores. The textual phrase `Apple MPS Backend` in previous reports was a static markdown header label and did not reflect the true runtime device, which was CPU.

---

## 8. FINAL SCIENTIFIC CLASSIFICATION & DECISION

Under Section 11 of the experimental protocol:

### Classification: **CATEGORY B / C: Positive but Inconclusive / No Demonstrated Incremental Value**

1. **Statistical Verdict**:
   * Participant-level paired bootstrap delta: $\mathbf{\Delta = +0.0044 \text{ AUROC}}$, 95% CI: $[-\mathbf{0.0029}, +\mathbf{0.0122}]$, $\mathbf{p = 0.2440}$.
   * Mean fold paired difference: $\mathbf{\Delta = +0.0077 \text{ AUROC}}$, $t = 1.6745$, $\mathbf{p = 0.1162}$.
   * In both evaluations, zero is contained within the confidence intervals ($p > 0.05$). The null hypothesis cannot be rejected.
2. **Contrast with FreeSurfer Morphometry**:
   * In the exact same cohort, adding FreeSurfer morphometry to clinical testing produces an increment of $\mathbf{+0.0433 \text{ AUROC}}$ (95% CI: $[+0.0135, +0.0747]$, $\mathbf{p = 0.0020}$). FreeSurfer provides robust, statistically significant predictive value.
   * Adding subcortical MRI ROI features on top of Clinical + FreeSurfer provides a negligible increment ($+0.0044$) that is statistically indistinguishable from zero.
3. **Clinical Conclusion**:
   * Anatomically informed subcortical MRI features do not overcome the saturation effect established by clinical cognitive testing and FreeSurfer morphometry.
   * This does not imply that MRI has no biological value; rather, the predictive information present in baseline structural MRI for a 24-month horizon is already fully captured by regional FreeSurfer morphometric summaries.
