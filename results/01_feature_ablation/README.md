# Feature Ablation and Sensitivity Analysis

This directory documents the comprehensive sensitivity analyses evaluating feature representations, follow-up censoring windows, diagnostic reversals, and non-circular feature subsets.

## Major Analyses and Findings

### 1. Strict Follow-Up Censoring Analysis ($N=583$)
- **Context:** In the primary cohort ($N=622$), 39 non-converters had their last documented follow-up visit between 640 and 729 days (within the standard 24-month clinical window).
- **Results:** Restricting the negative control group strictly to participants with documented negative visits at $\ge 730$ days ($N=583$, 144 converters, 439 non-converters) yields:
  - Clinical Only AUROC: **$0.8525$**
  - Clinical + FS29 AUROC: **$0.8860$** (95% CI: $[0.8555, 0.9129]$)
  - FreeSurfer Increment: $\Delta = \mathbf{+0.0335}$ (95% CI: $[+0.0070, +0.0595]$)
  - PR-AUC: **$0.7502$**
- **Conclusion:** Eliminating borderline censoring increases discrimination to $0.8860$ without altering the core scientific conclusion: FreeSurfer adds significant, reproducible predictive value.

### 2. Non-Circular Clinical Feature Sensitivity
- **Context:** To address potential circularity concerns regarding functional and cognitive dementia staging scales (CDR-SB, CDR Global, FAQ) that heavily correlate with the clinical diagnosis of AD, models were evaluated after removing these 3 features (retaining Age, Sex, Education, APOE4, MMSE, ADAS11, ADAS13).
- **Results:**
  - Non-Circular Clinical Only AUROC: **$0.8119$**
  - Non-Circular Clinical + FS29 AUROC: **$0.8615$** (95% CI: $[0.8254, 0.8917]$)
  - FreeSurfer Incremental Value: $\Delta = \mathbf{+0.0496}$ (95% CI: $[+0.0210, +0.0782]$)
- **Conclusion:** When diagnostic-adjacent scales are omitted, FreeSurfer morphometry provides an even larger incremental benefit (+0.0496 vs +0.0391), demonstrating that structural atrophy provides complementary biological signal.

### 3. Additional Robustness Checks
- **Exclusion of Early Converters ($\le 180$ days, $N=610$):** AUROC $0.8698$ ($\Delta = +0.0370$).
- **Diagnostic Reverter Exclusion ($N=547$):** AUROC $0.8655$ ($\Delta = +0.0519$).
- **Extended Follow-Up Horizon (820 days, $N=622$):** AUROC $0.8762$ ($\Delta = +0.0376$).
- **Cohort Subgroups:** ADNI1-only AUROC $0.8142$; ADNIGO/ADNI2-only AUROC $0.9026$.

For full tabular results, see [`metrics.csv`](file:///Users/lahari/Desktop/MCI_AD_Research/results/01_feature_ablation/metrics.csv) and [`experiment_summary.json`](file:///Users/lahari/Desktop/MCI_AD_Research/results/01_feature_ablation/experiment_summary.json).
