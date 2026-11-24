# Comprehensive Experimental Synthesis and Cross-Study Comparison

This directory synthesizes all empirical evidence generated across the project to answer the central research question:

> **Does baseline 3D MRI provide statistically defensible incremental predictive value beyond clinical measures and FreeSurfer morphometry for 24-month MCI-to-AD progression?**

## Master Synthesis Table

| Experiment & Approach | Evaluated Cohort | Positive / Total | Reference AUROC | Experimental AUROC | Incremental Delta ($\Delta$) | 95% Bootstrap CI | Empirical $p$-value | Statistical Conclusion |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline: FreeSurfer Increment** | Full Dev ($N=622$) | 144 / 622 | 0.8440 (Clin) | 0.8787 (Clin+FS) | **+0.0391** | $[+0.0154, +0.0638]$ | $0.0010$ | **Statistically Significant** |
| **Strict Follow-up ($\ge 730$d)** | Strict Dev ($N=583$) | 144 / 583 | 0.8525 (Clin) | 0.8860 (Clin+FS) | **+0.0335** | $[+0.0070, +0.0595]$ | $0.0020$ | **Statistically Significant** |
| **Whole-Volume 3D MRI Ablation** | Full Dev ($N=622$) | 144 / 622 | 0.8517 (Model A) | 0.8587 (Model B) | **+0.0070** | $[-0.0057, +0.0197]$ | $0.2860$ | **Not Significant** |
| **Subcortical MRI ROI Contrast** | Matched Dev ($N=564$) | 123 / 564 | 0.8643 (Model A) | 0.8688 (Model B) | **+0.0044** | $[-0.0029, +0.0122]$ | $0.2440$ | **Not Significant** |
| **Subcortical FS Validation** | Matched Dev ($N=564$) | 123 / 564 | 0.8210 (Clin) | 0.8643 (Clin+FS) | **+0.0433** | $[+0.0135, +0.0747]$ | $0.0020$ | **Statistically Significant** |

---

## Detailed Scientific Insights

### 1. Dual Independent Disconfirmation of MRI Incremental Value
Two radically different engineering paradigms were evaluated to extract information from baseline 3D T1 MRI:
1. **Whole-Volume 3D CNN with Cross-Modal Attention:** End-to-end representation learning across the whole skull-stripped volume ($N=622$), yielding $\Delta = +0.0070$, $p = 0.2860$.
2. **Anatomical Subcortical MRI ROI Contrast:** Targeted statistical texture features extracted from 16 Harvard-Oxford subcortical structures on a complete-case matched cohort ($N=564$), yielding $\Delta = +0.0044$, $p = 0.2440$.

Both paradigms converged on the same result: neither raw whole-volume 3D CNN representations nor targeted subcortical ROI intensity features yielded a statistically significant improvement over Clinical + FreeSurfer morphometry.

### 2. Consistency of the FreeSurfer Morphometric Increment
Across all tested cohorts and model architectures, FreeSurfer regional morphometry consistently provided robust, statistically significant prognostic improvement over clinical/cognitive scores alone:
- Primary 3-repeat ensemble ($N=622$): $\Delta = +0.0391$ (95% CI: $[+0.0154, +0.0638]$, $p = 0.0010$).
- Strict follow-up cohort ($N=583$): $\Delta = +0.0335$ (95% CI: $[+0.0070, +0.0595]$, $p = 0.0020$).
- Matched complete-case cohort ($N=564$): $\Delta = +0.0433$ (95% CI: $[+0.0135, +0.0747]$, $p = 0.0020$).

### 3. Nuanced Biological and Methodological Interpretation
These empirical findings must not be misinterpreted as claiming that MRI has no biological value or that FreeSurfer completely captures all information in MRI. Rather:
- Structural MRI provides critical information about cerebral atrophy and neurodegenerative changes.
- In the evaluated cohort and experimental settings, neither whole-volume MRI representations nor anatomically informed subcortical MRI ROI features demonstrated a statistically significant incremental predictive improvement beyond clinical and FreeSurfer morphometric features.
- We strictly preserve the distinction between failing to reject the null hypothesis of equal performance and claiming proof of equivalence or asserting that MRI cannot improve MCI-to-AD prediction under other designs.

For complete numerical artifacts, see [`metrics.csv`](file:///Users/lahari/Desktop/MCI_AD_Research/results/04_final_comparison/metrics.csv) and [`experiment_summary.json`](file:///Users/lahari/Desktop/MCI_AD_Research/results/04_final_comparison/experiment_summary.json).
