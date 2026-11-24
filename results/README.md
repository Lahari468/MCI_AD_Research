# Experimental Results and Benchmarks Directory

This directory contains the complete, reproducible experimental results, metrics, figures, and statistical evaluations for the project:

> **Evaluating the Incremental Predictive Value of Baseline 3D MRI Beyond Clinical and FreeSurfer Morphometry for 24-Month MCI-to-Alzheimer's Disease Progression.**

---

## Directory Organization

The results are organized chronologically and methodologically into dedicated subdirectories:

```
results/
├── 00_baselines/              # Reference clinical, FreeSurfer, and combined baselines
├── 01_feature_ablation/        # Sensitivity analyses (censoring horizons, non-circular features)
├── 02_whole_volume_3d_mri/     # Whole-Volume 3D MRI Ablation (3D CNN + cross-modal attention)
├── 03_subcortical_mri_roi/     # Anatomical Subcortical MRI ROI Incremental Experiment
├── 04_final_comparison/        # Cross-methodological synthesis and comparative master tables
├── figures/                   # Forest plots, missingness curves, and diagnostic figures
│
├── experiments/               # Stage 5 audit artifacts (preserved for automated regression test suite)
└── paper/                     # Manuscript summary tables and reference figures
```

---

## Key Methodological Distinctions

To ensure accurate interpretation and prevent methodological conflation, readers and researchers should note the following critical distinctions:

### 1. Cohort Distinctions: Full Development Cohort ($N=622$) vs Matched Complete-Case Cohort ($N=564$)
- **Full Development Cohort ($N=622$, 144 converters, 478 non-converters):**
  - Used in `00_baselines`, `01_feature_ablation`, and `02_whole_volume_3d_mri`.
  - All 622 participants have complete Clinical and FreeSurfer tabular data.
  - 564 participants have baseline 3D T1 MRI scans; 58 participants have missing MRI.
  - In `02_whole_volume_3d_mri`, missing MRI scans are zero-masked and handled via explicit modality availability indicators and attention dropout ($p=0.3$).
- **Matched Complete-Case Cohort ($N=564$, 123 converters, 441 non-converters):**
  - Used in `03_subcortical_mri_roi`.
  - Exactly restricts analysis to participants who have verified baseline 3D T1 MRI scans.
  - Eliminates all missingness imputation, masking, or indicator variables, providing an unambiguous matched comparison.
- **Strict Follow-Up Subgroup ($N=583$, 144 converters, 439 non-converters):**
  - Evaluated in `01_feature_ablation`.
  - Excludes 39 non-converters censored between 640 and 729 days to test sensitivity to the 24-month horizon.

### 2. Metric Distinctions: Mean-Repeat AUROC vs Pooled Repeat-Averaged AUROC
Across multi-repeat cross-validation protocols, two distinct summary statistics are reported:
- **Repeat Mean AUROC $\mathrm{AUROC}_{\mathrm{rep\_mean}} \pm \mathrm{SD}$:** The arithmetic mean of the fold-averaged AUROCs computed separately within each repeat.
- **Pooled Repeat-Averaged AUROC ($\text{AUROC}_{\text{pooled}}$):** The AUROC computed once across all participants after averaging their predicted out-of-fold probabilities across all repeats.
- **Why They Differ:** Averaging predicted probabilities across seeds smooths out fold boundary variance, typically yielding slightly higher discrimination (e.g., $0.876 \pm 0.006$ repeat mean vs **$0.8787$** pooled repeat-averaged AUROC for Clinical + FS29 on $N=622$). Both metrics are reported in all summary tables for full transparency.

### 3. Distinct MRI Methodologies: Whole-Volume 3D CNN vs Subcortical MRI ROIs
The two MRI branches represent fundamentally distinct computer vision paradigms and must not be conflated:
1. **Whole-Volume 3D MRI Ablation (`02_whole_volume_3d_mri`):**
   - End-to-end 3D CNN feature extractor trained directly on whole brain volumes.
   - Evaluated on $N=622$ using cross-modal attention fusion.
   - Result: Model A (0.8517) vs Model B (0.8587), $\Delta = +0.0070$, $p = 0.2860$.
2. **Anatomical Subcortical MRI ROI Experiment (`03_subcortical_mri_roi`):**
   - Targeted intensity feature extraction from 16 Harvard-Oxford subcortical ROIs.
   - Evaluated on $N=564$ using ridge-regularized feature fusion.
   - Result: Model A (0.8643) vs Model B (0.8688), $\Delta = +0.0044$, $p = 0.2440$.

---

## Master Result Summary

| Experiment Directory | Model / Comparison | Cohort ($N$) | Metric | Estimate (95% CI) | Paired $p$-value |
| :--- | :--- | :---: | :---: | :---: | :---: |
| [`00_baselines/`](file:///Users/lahari/Desktop/MCI_AD_Research/results/00_baselines/) | Clinical-Only L2 Logistic Regression | 622 | AUROC | 0.8440 [0.8100, 0.8780] | — |
| [`00_baselines/`](file:///Users/lahari/Desktop/MCI_AD_Research/results/00_baselines/) | FreeSurfer-Only Morphometry | 622 | AUROC | 0.8131 [0.7704, 0.8492] | — |
| [`00_baselines/`](file:///Users/lahari/Desktop/MCI_AD_Research/results/00_baselines/) | Clinical + FS29 (3-Repeat Ensemble) | 622 | AUROC | **0.8787** [0.8464, 0.9050] | — |
| [`01_feature_ablation/`](file:///Users/lahari/Desktop/MCI_AD_Research/results/01_feature_ablation/) | Strict Follow-up ($\ge 730$d) Clin+FS | 583 | AUROC | **0.8860** [0.8555, 0.9129] | — |
| [`01_feature_ablation/`](file:///Users/lahari/Desktop/MCI_AD_Research/results/01_feature_ablation/) | Non-Circular Features Clin+FS | 622 | AUROC | 0.8615 [0.8254, 0.8917] | — |
| [`02_whole_volume_3d_mri/`](file:///Users/lahari/Desktop/MCI_AD_Research/results/02_whole_volume_3d_mri/) | Whole-Volume 3D MRI Ablation ($\Delta$) | 622 | $\Delta$ AUROC | **+0.0070** [-0.0057, +0.0197] | 0.2860 (NS) |
| [`03_subcortical_mri_roi/`](file:///Users/lahari/Desktop/MCI_AD_Research/results/03_subcortical_mri_roi/) | Subcortical MRI ROI Contrast ($\Delta$) | 564 | $\Delta$ AUROC | **+0.0044** [-0.0029, +0.0122] | 0.2440 (NS) |
| [`03_subcortical_mri_roi/`](file:///Users/lahari/Desktop/MCI_AD_Research/results/03_subcortical_mri_roi/) | FreeSurfer Incremental Contrast ($\Delta$) | 564 | $\Delta$ AUROC | **+0.0433** [+0.0135, +0.0747] | **0.0020** |
| [`04_final_comparison/`](file:///Users/lahari/Desktop/MCI_AD_Research/results/04_final_comparison/) | Cross-Method Synthesis & Meta-Review | Multiple | — | Comprehensive master table | — |

---

## Scientific Conclusion

In the evaluated cohort and experimental settings, neither whole-volume MRI representations nor anatomically informed subcortical MRI ROI features demonstrated a statistically significant incremental predictive improvement beyond clinical and FreeSurfer morphometric features. We preserve the distinction between failing to reject the null hypothesis of equal performance and claiming proof of equivalence or stating that MRI has no biological value.

---

## Preserved Legacy Directories for Automated Tests

- `results/experiments/`: Preserves intermediate out-of-fold predictions (`e1_oof.csv`), fold splits (`e1_folds.csv`), permutation null distribution arrays (`*.npy`), and switch curves (`e2_switch_curve.csv`). These files are verified by unit tests in `tests/test_experiments_outputs.py`.
- `results/paper/`: Preserves tables and figures formatted for publication manuscripts (`T10a_mri_incremental_value.csv`, `T10b_mri_only_models.csv`, `F_forest_mri.png`). These are verified by unit tests in `tests/test_mri_tables.py`.
