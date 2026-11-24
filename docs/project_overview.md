# Project Overview

## Study Title
**Evaluating the Incremental Predictive Value of Baseline 3D MRI Beyond Clinical and FreeSurfer Morphometry for 24-Month MCI-to-Alzheimer's Disease Progression**

## Academic Context
- **Type**: B.Tech Undergraduate Research Opportunities Programme (UROP) Project
- **Host Institution**: Department of Computer Science & Engineering
- **Target Outcome**: Progression from Mild Cognitive Impairment (MCI) to Alzheimer's Disease Dementia within a 24-month (730-day) follow-up window.

## Abstract Summary
Structural magnetic resonance imaging (MRI) is widely utilized as a biomarker for neurodegenerative disease. However, MRI acquisition is costly, operationally intensive, and often difficult to deploy in community or primary care settings. 

This research investigates whether baseline structural 3D T1-weighted MRI provides statistically significant incremental predictive value beyond readily accessible clinical assessments (cognitive batteries, functional scales, APOE genotype) and regional FreeSurfer summary morphometry for predicting 24-month progression from MCI to Alzheimer's Disease.

Through rigorous multi-repeat cross-validation ($3 \times 5$ folds) and paired bootstrap evaluations on verified development cohorts ($N = 622$ and $N = 564$ matched complete-case participants from the Alzheimer's Disease Neuroimaging Initiative):
1. **Clinical + FreeSurfer vs. Clinical Only**: Adding regional morphometry produces a statistically significant improvement ($\Delta \text{AUROC} = +0.0433$, $95\%$ CI $[+0.0135, +0.0747]$, $p = 0.0020$).
2. **Whole-Volume 3D MRI Ablation (Stage 12)**: Adding a deep 3D convolutional neural network branch to clinical and FreeSurfer features yields $\Delta \text{AUROC} = +0.0070$ ($95\%$ CI $[-0.0057, +0.0197]$, $p = 0.2860$), failing to achieve statistical significance.
3. **Subcortical MRI ROI Contrast (Stage 19)**: Adding anatomically motivated subcortical tissue contrast features yields $\Delta \text{AUROC} = +0.0044$ ($95\%$ CI $[-0.0029, +0.0122]$, $p = 0.2440$), also failing to achieve statistical significance.

These findings demonstrate that while structural MRI is biologically informative and clinically essential for differential diagnosis, the predictive signal available in baseline MRI over a 24-month horizon is already captured by clinical cognitive scales and FreeSurfer morphometry.
