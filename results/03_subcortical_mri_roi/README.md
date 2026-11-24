# Anatomical Subcortical MRI ROI Incremental Experiment

This directory documents the 3-repeat, 5-fold cross-validation experiment testing whether targeted, anatomically informed intensity features from subcortical regions of interest (ROIs) add incremental prognostic value beyond standardized Clinical features and FreeSurfer morphometry.

## Rationale and Biological Motivation

While deep 3D CNNs process whole brain volumes end-to-end, they risk learning spurious non-biological correlations (e.g., scanner background noise, acquisition artifacts). Here, feature extraction is restricted strictly to 16 biologically validated subcortical structures from the Harvard-Oxford Subcortical Structural Atlas known to exhibit early Alzheimer's disease pathology (hippocampus, amygdala, thalamus, lateral ventricles, caudate, putamen, pallidum, accumbens).

To ensure complete fairness and eliminate any missing-data confounding, this experiment was conducted on the **matched complete-case development cohort ($N=564$)** where all participants have verified 3D T1 MRI scans.

## Experimental Setup

- **Cohort:** Matched complete-case cohort of 564 participants (124 converters, 440 non-converters; conversion rate: 21.99%).
- **Feature Extraction:** 6 robust intensity statistics (mean, standard deviation, 5th percentile, 95th percentile, median, IQR) extracted per ROI from skull-stripped, MNI152-registered T1 volumes.
- **Dimensionality Reduction:** Ridge logistic regression and PCA inside inner folds to prevent data leakage.
- **Biological Redundancy Check:** Pearson correlation between subcortical ROI intensity features and corresponding FreeSurfer volume features showed modest correlation ($r \approx 0.30 - 0.36$), confirming that tissue intensity contrast is non-identical to macroscopic volumetric atrophy.
- **Protocol:** Identical 3-repeat 5-fold cross-validation (15 folds per model family, seeds 42, 123, 456) using paired bootstrap hypothesis testing ($B=2000$).

## Primary Findings

| Model Configuration | AUROC | 95% CI | PR-AUC | Brier | $\Delta$ vs Model A | $p$-value | Significance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clinical Only Baseline** | 0.8210 | $[0.7760, 0.8624]$ | 0.5488 | 0.1342 | -0.0433 | 0.0020 | **Significant** (favors Model A) |
| **Model A: Clinical + FS29** | **0.8643** | $[0.8260, 0.8992]$ | 0.6598 | 0.1132 | Reference | — | — |
| **Model B: Clinical + FS29 + 16 ROIs** | **0.8688** | $[0.8312, 0.9026]$ | 0.6756 | 0.1116 | **+0.0044** | **0.2440** | **Not Significant** |
| **Model C: Target ROIs + Control ROIs** | 0.8640 | $[0.8251, 0.8998]$ | 0.6671 | 0.1140 | -0.0003 | 0.8770 | Not Significant |
| **Model B (Nonlinear GBM)** | 0.8616 | $[0.8214, 0.8984]$ | 0.6655 | 0.1140 | -0.0028 | 0.7560 | Not Significant |
| **MRI ROI Only Baseline** | 0.7180 | $[0.6650, 0.7680]$ | 0.4120 | 0.1530 | -0.1463 | <0.0001 | Not Significant |

### Statistical Contrast Analysis

1. **Incremental Value of Subcortical ROIs (Model B minus Model A):**
   - $\Delta \text{AUROC} = \mathbf{+0.0044}$ (95% CI: $[-0.0029, +0.0122]$, empirical bootstrap $p = \mathbf{0.2440}$).
   - The 95% bootstrap confidence interval crosses zero; the null hypothesis cannot be rejected.
2. **Incremental Value of FreeSurfer (Model A minus Clinical Baseline):**
   - $\Delta \text{AUROC} = \mathbf{+0.0433}$ (95% CI: $[+0.0135, +0.0747]$, empirical bootstrap $p = \mathbf{0.0020}$).
   - The addition of FreeSurfer morphometry to clinical variables is statistically significant.
3. **Negative Control Specificity:**
   - Incorporating non-AD control ROIs (precentral gyrus, brainstem) yielded zero improvement ($\Delta = -0.0003, p = 0.8770$), confirming that feature addition does not mechanically inflate performance.

## Scientific Conclusion

Targeted anatomical subcortical MRI ROI features provide modest standalone discrimination (AUROC 0.7180), but when combined with standardized Clinical markers and FreeSurfer morphometry, their incremental contribution ($\Delta = +0.0044, p = 0.2440$) is not statistically significant. This corroborates the Whole-Volume 3D MRI Ablation finding: baseline MRI intensity features do not provide statistically defensible incremental prognostic value beyond FreeSurfer morphometry and clinical testing.

For full data tables, see [`metrics.csv`](file:///Users/lahari/Desktop/MCI_AD_Research/results/03_subcortical_mri_roi/metrics.csv), [`paired_bootstrap_deltas.csv`](file:///Users/lahari/Desktop/MCI_AD_Research/results/03_subcortical_mri_roi/paired_bootstrap_deltas.csv), [`phase_breakdown.csv`](file:///Users/lahari/Desktop/MCI_AD_Research/results/03_subcortical_mri_roi/phase_breakdown.csv), and [`fs_mri_correlation.csv`](file:///Users/lahari/Desktop/MCI_AD_Research/results/03_subcortical_mri_roi/fs_mri_correlation.csv).
