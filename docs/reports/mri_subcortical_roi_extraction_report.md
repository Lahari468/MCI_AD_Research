# Harvard-Oxford Subcortical Anatomical ROI Extraction Report

**Date**: November 23, 2026  
**Repository**: `.`  
**Output File**: `data/processed/features/mri_subcortical_roi_features.csv`  
**Scope**: 564 development cohort participants with complete, usable baseline 3D T1 MRI. Locked holdout ($N=156$) strictly excluded.

---

## 1. Reference Atlas Metadata & Alignment Verification

* **Atlas**: Harvard-Oxford Subcortical Structural Atlas (probabilistic, maximum probability thresholded at 25%).
* **Local File**: `data/reference/atlases/HarvardOxford-sub-maxprob-thr25-2mm.nii.gz`
* **Coordinate Space**: MNI152 Linear Standard Space, 2 mm isotropic voxel resolution.
* **Spatial Grid**: $(91 \times 109 \times 91)$ voxels.
* **Affine Matrix**:
  $$\begin{bmatrix} -2.0 & 0.0 & 0.0 & 90.0 \\ 0.0 & 2.0 & 0.0 & -126.0 \\ 0.0 & 0.0 & 2.0 & -72.0 \\ 0.0 & 0.0 & 0.0 & 1.0 \end{bmatrix}$$
* **Image Space Concordance**:
  All input volumes in `data/interim/mri_mni2mm_v2/<RID>.npy` were rigidly and affinely registered to `MNI152_T1_2mm.nii.gz` during preprocessing (`mri_preprocess_v2.py`) using FLIRT (12 DOF) with N4 bias-field correction and brain-mask intensity normalization. The voxel dimensions, orientation, and bounding box of the input tensors match the Harvard-Oxford atlas bit-for-bit.

---

## 2. Anatomical Region Mapping and Voxel Counts

The Harvard-Oxford Subcortical atlas defines 21 non-zero labels:

| Label Index | Anatomical Structure | Hemisphere | Target Role | Voxel Count (2 mm) | Volume ($\text{cm}^3$) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | Cerebral White Matter | Left | Intensity Reference | 31,371 | 250.97 |
| **12** | Cerebral White Matter | Right | Intensity Reference | 31,361 | 250.89 |
| **9** | Hippocampus | Left | Primary AD Target | 691 | 5.53 |
| **19** | Hippocampus | Right | Primary AD Target | 700 | 5.60 |
| **10** | Amygdala | Left | Primary AD Target | 306 | 2.45 |
| **20** | Amygdala | Right | Primary AD Target | 366 | 2.93 |
| **3** | Lateral Ventricle | Left | Primary AD Target | 1,145 | 9.16 |
| **14** | Lateral Ventricle | Right | Primary AD Target | 1,019 | 8.15 |
| **4** | Thalamus | Left | Control Structure | 1,321 | 10.57 |
| **15** | Thalamus | Right | Control Structure | 1,283 | 10.26 |
| **5** | Caudate | Left | Control Structure | 487 | 3.90 |
| **16** | Caudate | Right | Control Structure | 511 | 4.09 |
| **6** | Putamen | Left | Control Structure | 799 | 6.39 |
| **17** | Putamen | Right | Control Structure | 796 | 6.37 |

---

## 3. Cortical Region Feasibility & Limitations

* **Target Regions Requested by User**: Entorhinal cortex, parahippocampal gyrus, fusiform gyrus, inferior/middle temporal gyri, posterior cingulate, and precuneus.
* **Atlas Status**: **NOT AVAILABLE in the workspace**. The local atlas directory contains strictly the Harvard-Oxford Subcortical atlas. Cortical labels require surface-based reconstruction or dedicated non-linear cortical parcellations.
* **Defensible Alternative**: Cortical morphometry for these exact participants is already captured natively in subject-space by FreeSurfer and contained within the 29 FreeSurfer features (`data/processed/features/freesurfer29_raw_v1.csv`). As instructed by the scientific protocol, no artificial cortical atlas or synthetic transformation was invented. Subcortical extraction is conducted strictly on verified, validated subcortical targets.

---

## 4. Feature Extraction Formulation

In the preprocessed volumes, voxel intensities were globally brain-masked and z-scored:
$$X = \frac{x_{\text{clipped}} - \mu_{\text{brain}}}{\sigma_{\text{brain}}}$$

To eliminate inter-subject scanner gain drift and residual coil sensitivity variations, each regional statistic is referenced to the median intensity of bilateral cerebral white matter ($X_{\text{WM}}$, 62,732 total voxels):
$$\text{WM}_{\text{median}} = \text{median}(X_{\text{WM}})$$

For each region of interest $\text{ROI}$:
1. **Median Contrast**: $\tilde{I}_{50} = \text{median}(X_{\text{ROI}}) - \text{WM}_{\text{median}}$
2. **Mean Contrast**: $\tilde{I}_{\text{mean}} = \text{mean}(X_{\text{ROI}}) - \text{WM}_{\text{median}}$
3. **Robust Tails**: $\tilde{I}_{10} = p_{10}(X_{\text{ROI}}) - \text{WM}_{\text{median}}, \quad \tilde{I}_{90} = p_{90}(X_{\text{ROI}}) - \text{WM}_{\text{median}}$
4. **Internal Dispersion (Texture/Heterogeneity)**: $\sigma_{\text{ROI}} = \text{std}(X_{\text{ROI}})$
5. **Interquartile Spread**: $\text{IQR} = p_{75}(X_{\text{ROI}}) - p_{25}(X_{\text{ROI}})$
6. **Ventricle-to-Hippocampus Difference**: $\Delta_{\text{vent-hippo}} = \text{Ventricle}_{\text{bilateral, median}} - \text{Hippo}_{\text{bilateral, median}}$

---

## 5. Quality Control and Validation Results

* **Total Scans Processed**: Exactly 564 participants.
* **Extraction Success Rate**: 100% (564 / 564 `OK`).
* **Numerical Integrity**: Zero `NaN`, zero `Inf`, zero missing fields.
* **White Matter Stability**: White matter median intensity across the cohort exhibited tight clustering ($\mu = 1.054 \pm 0.082$), confirming high quality of prior N4 bias correction and intensity normalization.
* **Anatomical Plausibility**:
  * Lateral ventricle intensities were uniformly negative ($\approx -2.5\text{ to } -3.0$ relative to white matter), representing low CSF signal on T1.
  * Hippocampal and amygdaloid gray matter intensities were consistently intermediate ($\approx -1.0\text{ to } -1.2$ relative to white matter).
  * Caudate and putamen clustered at $\approx -0.5\text{ to } -0.7$.
