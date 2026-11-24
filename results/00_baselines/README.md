# Baseline Model Benchmarks

This directory archives the reference non-deep learning baseline experiments evaluated on the primary development cohort ($N=622$, 144 converters, 478 non-converters) under rigorous 5-fold cross-validation without test set contamination.

## Overview of Baseline Models

1. **Clinical-Only Logistic Regression (10 Features):**
   - **Features:** Age, sex, education, APOE4 carrier status, MMSE, CDR-SB, CDR Global, FAQ, ADAS11, ADAS13.
   - **Performance:** Single-repeat CV achieved AUROC $0.8426$. Across a 3-repeat 5-fold CV protocol, the repeat mean AUROC is $0.834 \pm 0.018$ and pooled repeat-averaged AUROC is $0.8440$ (95% CI: $[0.8100, 0.8780]$).
   - **Key Finding:** Clinical and cognitive scores establish a strong predictive anchor for 24-month progression.

2. **FreeSurfer-Only Morphometry (29 Features):**
   - **Features:** 29 cortical and subcortical morphometric volumetric and thickness features derived from FreeSurfer and normalized by intracranial volume (`ST10CV`).
   - **Performance:** 5-fold CV AUROC $0.8131$ (95% CI: $[0.7704, 0.8492]$), PR-AUC $0.5958$, Brier score $0.1345$.
   - **Key Finding:** FreeSurfer morphometry alone provides substantial prognostic signal, but lags clinical/cognitive performance.

3. **Combined Clinical + FreeSurfer FS29 Benchmark:**
   - **Initial Single 5-Fold Cross-Validation:** Achieved AUROC $0.8632$ (PR-AUC $0.6690$, Brier $0.1160$) on Repeat 0.
   - **Primary 3-Repeat 5-Fold CV Ensemble (15 Models):** Averaging predicted probabilities across 3 distinct partition seeds (42, 123, 456) yielded a repeat mean AUROC of $0.876 \pm 0.006$ and a pooled repeat-averaged AUROC of **$0.8787$** (95% CI: $[0.8464, 0.9050]$), PR-AUC $0.7088$, Brier score $0.1120$.
   - **Key Finding:** Combining FreeSurfer morphometry with clinical markers yields a statistically significant gain over clinical alone ($\Delta = +0.0391$, 95% CI: $[+0.0154, +0.0638]$).

## Metrics Summary

See [`metrics.csv`](file:///Users/lahari/Desktop/MCI_AD_Research/results/00_baselines/metrics.csv) and [`experiment_summary.json`](file:///Users/lahari/Desktop/MCI_AD_Research/results/00_baselines/experiment_summary.json) for machine-readable performance metrics and configuration metadata.
