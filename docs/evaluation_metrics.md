# Evaluation Metrics & Empirical Results

## Primary & Secondary Metrics
- **AUROC (Area Under the Receiver Operating Characteristic)**: Measures global ranking discrimination across all thresholds (primary metric).
- **PR-AUC (Precision-Recall Area / Average Precision)**: Focuses on positive conversion detection in class-imbalanced settings.
- **Brier Score**: Measures probability calibration error:
  $$\text{Brier} = \frac{1}{N} \sum_{i=1}^{N} (p_i - y_i)^2$$
- **Operating Metrics**: Evaluated at Youden's $J$-optimized threshold (Accuracy, Balanced Accuracy, Sensitivity, Specificity, Precision, F1-Score).

---

## Benchmark Empirical Results Summary

### 1. Stage 12 Whole-Volume 3D MRI Ablation ($N = 622$, 3 Repeats $\times$ 5 Folds)
- **Model A (Clinical 10 + FS29 Attention, MRI Excluded)**:
  - Pooled Repeat-Averaged AUROC: **0.8517**
- **Model B (Clinical 10 + FS29 + 3D MRI Attention)**:
  - Pooled Repeat-Averaged AUROC: **0.8587**
- **Paired Delta ($\text{Model B} - \text{Model A}$)**:
  - $\Delta \text{AUROC} = \mathbf{+0.0070}$, $95\%$ CI: $[\mathbf{-0.0057}, \mathbf{+0.0197}]$, empirical $p = \mathbf{0.2860}$ (Not Significant).

---

### 2. Stage 19 Subcortical MRI ROI Incremental Value ($N = 564$, 3 Repeats $\times$ 5 Folds)
- **Clinical Only (10 Features)**:
  - Pooled Repeat-Averaged AUROC: **0.8210**, PR-AUC: 0.5488, Brier: 0.1342
- **Model A (Clinical 10 + FS29)**:
  - Pooled Repeat-Averaged AUROC: **0.8643**, PR-AUC: 0.6598, Brier: 0.1132
- **Model B (Clinical 10 + FS29 + 16 Subcortical ROIs)**:
  - Pooled Repeat-Averaged AUROC: **0.8688**, PR-AUC: 0.6756, Brier: 0.1116
- **Model C (Clinical 10 + FS29 + 16 Target + 6 Control ROIs)**:
  - Pooled Repeat-Averaged AUROC: **0.8640**, PR-AUC: 0.6671, Brier: 0.1139
- **MRI ROI Only (16 Features Alone)**:
  - Pooled Repeat-Averaged AUROC: **0.7183**, PR-AUC: 0.4079, Brier: 0.1534
- **Model B Non-linear Gradient Boosting**:
  - Pooled Repeat-Averaged AUROC: **0.8616**, PR-AUC: 0.6655, Brier: 0.1139

### Paired Contrasts ($N = 564, B = 2,000$):
- **Model B vs. Model A (Primary)**:
  - $\Delta \text{AUROC} = \mathbf{+0.0044}$, $95\%$ CI: $[\mathbf{-0.0029}, \mathbf{+0.0122}]$, $p = \mathbf{0.2440}$ (Not Significant).
- **Model A vs. Clinical Only (Baseline Validation)**:
  - $\Delta \text{AUROC} = \mathbf{+0.0433}$, $95\%$ CI: $[\mathbf{+0.0135}, \mathbf{+0.0747}]$, $p = \mathbf{0.0020}$ (Statistically Significant, $p < 0.01$).
