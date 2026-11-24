# Evaluating the Incremental Predictive Value of Baseline 3D MRI Beyond Clinical and FreeSurfer Morphometry for 24-Month MCI-to-AD Progression

A reproducible empirical evaluation assessing whether baseline whole-volume 3D MRI or targeted subcortical ROI intensity features provide statistically significant incremental prognostic value beyond standardized clinical assessments and FreeSurfer morphometry for 24-month MCI-to-AD progression.


## 1. Overview

Predicting progression from Mild Cognitive Impairment (MCI) to Alzheimer's Disease (AD) dementia within 24 months is vital for clinical trial enrichment and early intervention. While multimodal deep learning frameworks frequently incorporate raw structural Magnetic Resonance Imaging (MRI) volumes, the incremental prognostic contribution of voxel-level representations over clinical testing and regional morphometry remains actively debated.

- **Research Question:** Does baseline 3D structural MRI provide statistically defensible incremental predictive value once standardized clinical scores and FreeSurfer morphometric features are already accounted for?
- **Modalities Evaluated:**
  1. *Clinical & Cognitive (10 features):* Demographics, APOE4 status, and cognitive/functional batteries.
  2. *FreeSurfer Morphometry (FS29):* Intracranial volume-normalized cortical thickness and subcortical volume measures.
  3. *Structural 3D T1 MRI:* Whole-volume 3D convolutional neural network (CNN) representations and 16 targeted subcortical ROI-derived intensity/heterogeneity features across primary AD-vulnerable regions (hippocampus, amygdala, lateral ventricles).


## 2. Research Objective

Multimodal deep learning literature often assumes that adding 3D MRI volumes to clinical predictors improves prognostic accuracy. However, volumetric image representations introduce substantial model complexity, computational overhead, and risk of overfitting.

This project investigates whether baseline MRI features—as whole-brain 3D volumes or targeted subcortical intensity distributions—provide a statistically significant improvement in discrimination and calibration beyond properly regularized tabular clinical and morphometric models via leak-free multi-repeat cross-validation and participant-level paired bootstrap hypothesis testing.


## 3. Methodology

### a. Cohort Construction & Verification
From 1,199 screened ADNI participants, 778 met inclusion criteria: baseline MCI, complete clinical scales, available FreeSurfer morphometry, and documented 24-month outcomes (conversion $\le 730$ days; non-conversion follow-up $\ge 640$ days).
- **Primary Development Cohort ($N = 622$):** 144 converters (23.15%), 478 non-converters; 564 have baseline 3D T1 MRI (58 missing scans zero-masked with explicit indicators in whole-volume models).
- **Matched Complete-Case Cohort ($N = 564$):** 123 converters (21.81%), 441 non-converters. Restricts analysis to participants with complete baseline MRI for unmasked ROI evaluation.
- **Locked Internal Holdout ($N = 156$):** 28 converters, 128 non-converters. Preserved untouched for final validation.

Details and flowcharts: [docs/dataset_and_cohort.md](docs/dataset_and_cohort.md).

### b. Feature Sets
- **Clinical / Cognitive (10):** Age, sex, education, APOE4 count, MMSE, CDR-SB, CDR Global, FAQ, ADAS11, ADAS13.
- **FreeSurfer Morphometry (FS29):** 29 cortical thickness and subcortical volume features; volumes normalized by total intracranial volume (`ST10CV`).
- **Subcortical ROIs (16 Target Features):** Extracted from Harvard-Oxford subcortical atlas structures (bilateral hippocampus, amygdala, and lateral ventricles). Features include 9 median intensity contrast measures relative to reference white matter, 6 intensity dispersion/heterogeneity measures (standard deviation), and 1 composite ventricle-to-hippocampus contrast difference.

Details: [docs/feature_engineering.md](docs/feature_engineering.md).

### c. MRI Preprocessing & Quality Control
Baseline T1-weighted volumes were processed without native skull-stripping. The pipeline applied spatial reorientation (`fslreorient2std`), field-of-view cropping (`robustfov`), N4 bias-field correction (SimpleITK), and 12-DOF affine registration (`flirt`) to the MNI152 2mm template ($91 \times 109 \times 91$ voxels). Post-registration brain extraction and intensity z-scoring were performed using the standard MNI152 template brain mask. Quality control audited registration alignment using normalized cross-correlation (NCC) to the MNI152 template inside the brain mask, affine scaling, and rotation angle thresholds. Scanner metadata (e.g., 1.5T vs 3.0T field strength) was cataloged across cohorts without serving as an automated quality filter. Details: [docs/mri_preprocessing.md](docs/mri_preprocessing.md).

### d. Model Architectures
- **Tabular Baselines:** L2-regularized Logistic Regression with nested validation for $C \in [10^{-3}, 10^{2}]$ and decision thresholds.
- **Multimodal Architecture (Stage 12):** Tabular MLP encoders and a 4-stage 3D CNN integrated via **inter-modality multi-head self-attention** ($h = 2, d = 32$) with missingness conditioning, trained on CPU.
- **Subcortical ROI Fusion (Stage 19):** L2-regularized logistic regression combining clinical, FreeSurfer, and subcortical ROI intensity features with inner-fold median imputation and z-score standardization (no PCA).

Details: [docs/model_architecture.md](docs/model_architecture.md).

### e. Evaluation Strategy
- **Cross-Validation:** 3-repeat 5-fold stratified cross-validation (fixed seeds 42, 123, 456) with fold-isolated preprocessing.
- **Metric Reporting:** Both fold-averaged repeat mean AUROC and pooled repeat-averaged AUROC (computed on participant-level out-of-fold probability vectors averaged across repeats) are tracked.
- **Statistical Inference:** Participant-level paired bootstrap hypothesis testing ($B = 2,000$ resamples) on out-of-fold predictions evaluates incremental differences ($\Delta$) in AUROC, PR-AUC, and Brier score. Details: [docs/experimental_design.md](docs/experimental_design.md).


## 4. Key Findings

The table below highlights the primary experimental comparisons. Metrics reflect pooled repeat-averaged AUROC across 3-repeat 5-fold cross-validation. Detailed confidence intervals, PR-AUC, Brier scores, subgroup analyses, and exploratory experiments are archived in [results/README.md](results/README.md) and [docs/experiment_history.md](docs/experiment_history.md).

| Paradigm / Model | Evaluated Cohort | Sample Size ($N$) | Pooled AUROC | Incremental $\Delta$ AUROC | $p$-value | Scientific Conclusion |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Clinical Baseline** | Primary Dev | 622 | 0.8440 | Reference | — | Tabular Reference |
| **FreeSurfer Baseline** | Primary Dev | 622 | 0.8131 | -0.0309 | — | Morphometry Reference |
| **Clinical + FS29 (Primary)** | Primary Dev | 622 | **0.8787** | **+0.0391** | **0.0010** | **Statistically Significant** |
| **Whole-Volume 3D MRI (Stage 12)** | Primary Dev | 622 | 0.8517 $\rightarrow$ **0.8587** | **+0.0070** | **0.2860** | **Not Significant** (Model B vs A) |
| **Subcortical MRI ROIs (Stage 19)** | Matched Dev | 564 | 0.8643 $\rightarrow$ **0.8688** | **+0.0044** | **0.2440** | **Not Significant** (Model B vs A) |
| **Strict Follow-up ($\ge 730$d)** | Strict Dev | 583 | **0.8860** | **+0.0335** | **0.0020** | **Statistically Significant** (Sensitivity) |

### Core Conclusions
1. **FreeSurfer Morphometry Adds Value:** FreeSurfer morphometric features provide statistically significant predictive improvement over clinical testing alone ($\Delta \text{AUROC} = +0.0391, p = 0.0010$ on $N=622$; $\Delta \text{AUROC} = +0.0433, p = 0.0020$ on $N=564$).
2. **Dual Disconfirmation of MRI Incremental Benefit:** Neither whole-volume 3D CNN embeddings ($\Delta \text{AUROC} = +0.0070, p = 0.2860$) nor targeted subcortical ROI intensity features ($\Delta \text{AUROC} = +0.0044, p = 0.2440$) demonstrated a statistically significant gain beyond Clinical + FreeSurfer morphometry.
3. **Defensible Scientific Conclusion:** In the evaluated cohort and experimental settings, neither whole-volume MRI representations nor anatomically informed subcortical MRI ROI features demonstrated a statistically significant incremental predictive improvement beyond clinical and FreeSurfer morphometric features. We preserve the critical distinction between failing to reject the null hypothesis of equal performance and claiming proof of equivalence or asserting that MRI lacks biological value.
4. **Tissue Intensity vs Volumetric Morphometry:** Correlation between subcortical ROI intensity measures and FreeSurfer volumes is modest ($r = 0.30 - 0.36$). While this indicates an association between local intensity measures and macroscopic volumetric morphometry, it does not establish biological equivalence, independence, or a causal explanation for the absence of incremental predictive gain.

For complete chronological experiment reports, see [docs/experiment_history.md](docs/experiment_history.md) and [results/README.md](results/README.md).


## 5. Repository Structure

```
MCI_AD_Research/
├── docs/                        # Methodological specifications and experiment history
├── configs/                     # Centralized YAML/JSON experiment configurations
├── src/                         # Modular library (data, features, models, training, eval)
├── scripts/                     # Standalone execution drivers (data prep, training, eval)
├── results/                     # Chronological results archive (00_baselines to 04_final)
├── examples/                    # Standalone synthetic demonstration pipeline
├── tests/                       # 55 automated unit and regression tests
├── requirements.txt             # Pinned package dependencies
├── LICENSE                      # Academic research license
└── CITATION.cff                 # Machine-readable citation metadata
```

Detailed documentation and result tables are available in [`docs/`](docs/) and [`results/`](results/).


## 6. Reproducibility & Setup

### Environment Setup
The codebase is tested and verified on **Python 3.9** (specifically Python 3.9.6):

```bash
git clone https://github.com/Lahari468/MCI_AD_Research.git
cd MCI_AD_Research
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest tests/
```

### Self-Contained Synthetic Demonstration
Researchers can verify data loading, model forward passes, self-attention fusion, and paired bootstrap evaluation without requiring ADNI credentials:

```bash
python examples/synthetic_data_example/run_synthetic_pipeline.py
```

This script generates synthetic tabular and 3D volumetric data, fits baseline and multimodal models, and computes paired bootstrap statistics in ~3 seconds.

### Full ADNI Reproduction Prerequisites
Full reproduction from raw ADNI scans requires credentialed ADNI access, downloaded imaging volumes, and neuroimaging software (FSL 6.0+). The modular pipeline covers cohort construction, feature extraction, 3D affine registration, baseline models, multimodal CPU training, and paired bootstrap statistical testing. Full stage-by-stage prerequisites and validation levels are documented in [docs/reproducibility.md](docs/reproducibility.md).


## 7. Data Access & Privacy

To strictly comply with the **ADNI Data Use Agreement** and protect participant privacy, this repository contains zero Protected Health Information, participant identifiers (`RID`, subject IDs, exam dates), raw clinical CSVs, imaging volumes (DICOM/NIfTI), or trained neural network weights.

Researchers must obtain credentialed access directly via the [LONI ADNI Portal](https://adni.loni.usc.edu/). Once access is approved and raw files are downloaded, local data paths must be configured in [`configs/paths.yaml`](configs/paths.yaml) to run raw-data pipeline scripts.


## 8. License & Citation

### Academic Research License
This codebase is released under a custom **Academic Research Code License** for non-commercial academic research and educational evaluation; it is not an unrestricted open-source license and distributes no ADNI data. See [`LICENSE`](LICENSE) for complete terms.

### Citation
This repository contains an unpublished academic research codebase. If you reference this work or methodology, cite:

```bibtex
@misc{kadiyala2026mci,
  author = {Kadiyala, Jaya Lahari},
  title = {Evaluating the Incremental Predictive Value of Baseline 3D MRI Beyond Clinical and FreeSurfer Morphometry for 24-Month MCI-to-Alzheimer's Disease Progression},
  year = {2026},
  howpublished = {\url{https://github.com/Lahari468/MCI_AD_Research}},
  note = {Unpublished academic research repository}
}
```

Machine-readable bibliographic metadata is available in [`CITATION.cff`](CITATION.cff).


## 9. Acknowledgments

Data collection and sharing for this research was facilitated by the Alzheimer's Disease Neuroimaging Initiative (ADNI). Investigators within the ADNI contributed to the design and implementation of ADNI and provided data, but did not participate in the specific statistical analysis or writing of this report. We express our gratitude to the study participants and their families across all ADNI participating sites.
