# Whole-Volume 3D MRI Ablation Experiment

This directory documents the 3-repeat, 5-fold cross-validation ablation experiment assessing whether adding raw whole-volume 3D MRI features via a 3D Convolutional Neural Network (CNN) provides incremental prognostic value beyond standardized Clinical and FreeSurfer morphometry features.

## Experimental Design and Methodology

- **Cohort:** Primary development cohort of 622 participants (144 converters, 478 non-converters), of whom 564 have baseline 3D T1-weighted MRI scans and 58 have missing MRI.
- **Handling MRI Missingness:** Participants without MRI scans have their MRI tensor zero-masked, coupled with an explicit binary availability indicator and attention dropout ($p=0.3$) to prevent the network from developing spurious dependencies on missingness artifacts.
- **Architecture:** Compact multimodal cross-modal attention network:
  - Tabular Clinical Encoder: Linear projection to $d=32$.
  - FreeSurfer Encoder: Linear projection to $d=32$.
  - 3D MRI Branch: 4-stage 3D ConvNet with spatial downsampling to 32-dimensional embedding (743,665 total parameters in Model B vs 106,753 in Model A).
  - Fusion: Multi-head cross-attention ($h=2$, $d=32$) with residual connections.
- **Protocol:** Identical 3-repeat 5-fold cross-validation (30 model runs: 15 for Model A, 15 for Model B) across fixed partition seeds (0, 1, 2) and identical initialization. Out-of-fold predictions were averaged across the 3 repeats.

## Empirical Findings

| Model | Modalities | Parameters | Repeat Mean AUROC | Pooled Repeat-Averaged AUROC | 95% CI | PR-AUC | Brier |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A (Control)** | Clinical + FS29 (No MRI) | 106,753 | $0.8332 \pm 0.0083$ | **$0.8517$** | $[0.8142, 0.8872]$ | 0.6640 | 0.1392 |
| **Model B (Experimental)** | Clinical + FS29 + 3D MRI | 743,665 | $0.8348 \pm 0.0107$ | **$0.8587$** | $[0.8219, 0.8936]$ | 0.6712 | 0.1354 |

### Paired Bootstrap Hypothesis Testing ($B=2000$)

- **AUROC Difference:** $\Delta = \mathbf{+0.0070}$ (95% CI: $[-0.0057, +0.0197]$), two-tailed empirical bootstrap $p = \mathbf{0.2860}$ (**Not Statistically Significant**).
- **PR-AUC Difference:** $\Delta = \mathbf{+0.0073}$ (95% CI: $[-0.0271, +0.0471]$), $p = \mathbf{0.7140}$ (Not Significant).
- **Brier Score Difference:** $\Delta = \mathbf{-0.0038}$ (95% CI: $[-0.0081, +0.0004]$), $p = \mathbf{0.0790}$ (Not Significant).

### Exploratory 3-Channel MRI Variation

An exploratory variant extracting intensity, local spatial mean, and local spatial variance channels via 3D average pooling was evaluated across the full cohort. It achieved a pooled AUROC of **$0.8450$** (95% CI: $[0.8090, 0.8840]$) and PR-AUC of $0.6290$, indicating that local first-order texture statistics did not outperform single-channel representations.

## Scientific Conclusion

The empirical paired bootstrap test fails to reject the null hypothesis of equal performance ($p = 0.2860$). Raw whole-volume 3D MRI processed through a standard 3D CNN does not provide a statistically significant incremental benefit over Clinical features and FreeSurfer morphometry alone in this 622-patient development cohort.

For detailed numbers, see [`metrics.csv`](file:///Users/lahari/Desktop/MCI_AD_Research/results/02_whole_volume_3d_mri/metrics.csv), [`paired_bootstrap_deltas.csv`](file:///Users/lahari/Desktop/MCI_AD_Research/results/02_whole_volume_3d_mri/paired_bootstrap_deltas.csv), and [`experiment_summary.json`](file:///Users/lahari/Desktop/MCI_AD_Research/results/02_whole_volume_3d_mri/experiment_summary.json).
