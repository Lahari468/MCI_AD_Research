# Stage 12: Three-Repeat Matched MRI Ablation Report

**Experiment Date**: 2026-11-23 22:09:15  
**Repository**: `.`  
**Device**: `cpu` (Apple MPS Backend)  
**Target Outcome**: 24-month progression from MCI to Alzheimer's Disease (`label_primary`)  
**Data Scope**: Strictly Development Participants ($N=622$, 144 converters). Locked holdout ($N=156$) completely excluded.

---

## 1. Experimental Setup and Architecture Matching

| Model Component | Model A (Two-Modality) | Model B (Three-Modality) |
| :--- | :---: | :---: |
| **Modalities Included** | Clinical + FreeSurfer FS29 | Clinical + FreeSurfer FS29 + Baseline 3D MRI |
| **Clinical Encoder** | MLPEnc: 10 -> 128 -> 256 -> 32 | MLPEnc: 10 -> 128 -> 256 -> 32 |
| **FreeSurfer Encoder** | MLPEnc: 29 -> 128 -> 256 -> 32 | MLPEnc: 29 -> 128 -> 256 -> 32 |
| **3D MRI Encoder** | EXCLUDED (`None`) | CNN3D: 1 -> 16 -> 32 -> 64 -> 96 -> 32 |
| **Total Trainable Parameters** | **106,753** | **743,665** |
| **Shared Weight Initialization** | 100% Identical Initial Weights | 100% Identical Initial Weights |
| **Optimization** | AdamW (lr=1e-4, wd=5e-3, bs=8) | AdamW (lr=1e-4, wd=5e-3, bs=8) |
| **Modality Dropout** | $p = 0.3$ on FS | $p = 0.3$ on FS and MRI |
| **Cross-Validation** | 3 Repeats x 5 Folds (15 fits) | 3 Repeats x 5 Folds (15 fits) |

---

## 2. Multi-Repeat Summary

| Repeat | Model A AUROC | Model B AUROC | Delta AUROC (B - A) | Model A PR-AUC | Model B PR-AUC | Delta PR-AUC |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Repeat 0** | 0.8303 | 0.8225 | -0.0077 | 0.6320 | 0.6304 | -0.0016 |
| **Repeat 1** | 0.8267 | 0.8401 | +0.0135 | 0.5998 | 0.6106 | +0.0108 |
| **Repeat 2** | 0.8426 | 0.8418 | -0.0008 | 0.6281 | 0.6454 | +0.0174 |
| **Mean ± SD** | **0.8332 ± 0.0083** | **0.8348 ± 0.0107** | **+0.0016** | **0.6200 ± 0.0176** | **0.6288 ± 0.0175** | **+0.0089** |

---

## 3. Primary Paired Participant-Level Bootstrap Evaluation ($N = 622, B = 2,000$)

Evaluating repeat-averaged out-of-fold predicted probabilities for each participant:

| Metric | Model A (No MRI) | Model B (+MRI) | Paired Delta (B - A) [95% CI] | Two-Sided p-value | Significance |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **AUROC** | **0.8517** | **0.8587** | **+0.0070** [-0.0057, +0.0197] | **p = 0.2860** | NOT SIGNIFICANT (p > 0.05) |
| **PR-AUC** | **0.6640** | **0.6712** | **+0.0073** [-0.0271, +0.0471] | **p = 0.7140** | NOT SIGNIFICANT (p > 0.05) |
| **Brier Score** | **0.1392** | **0.1354** | **-0.0038** [-0.0081, +0.0004] | **p = 0.0790** | NOT SIGNIFICANT (p > 0.05) |

---

## 4. Scientific Verdict

1. **Ablation Hypothesis:** If 3D MRI contains significant incremental biological signal beyond clinical testing and FreeSurfer summary morphometry, Model B should statistically outperform Model A.
2. **Empirical Finding:** Across 30 matched model-fold fits spanning 3 distinct random cross-validation repeats, the paired delta in AUROC is **+0.0070** (95% CI: [-0.0057, +0.0197], p = 0.2860).
3. **Conclusion:** Incremental predictive value of baseline 3D MRI beyond Clinical + FreeSurfer morphometry **was not demonstrated** under this controlled deep learning architecture.
