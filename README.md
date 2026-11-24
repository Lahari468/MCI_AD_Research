# Incremental Predictive Value of Baseline 3D MRI in MCI-to-AD Progression

Research codebase evaluating whether baseline structural 3D T1 MRI adds statistically significant incremental predictive power beyond clinical assessments and FreeSurfer morphometry for 24-month Mild Cognitive Impairment (MCI) to Alzheimer's Disease progression.

---

## 1. Research Overview

- **Objective**: Determine whether baseline 3D structural MRI provides incremental predictive value beyond clinical cognitive scales and FreeSurfer summary morphometry.
- **Target Event**: 24-Month (730-day) progression from MCI to Alzheimer's Disease dementia.
- **Cohorts**: Verified development cohort ($N = 622$ primary dev; $N = 564$ complete-case MRI dev) and locked internal holdout ($N = 156$) derived from the Alzheimer's Disease Neuroimaging Initiative (ADNI).
- **Evaluation Protocol**: Matched 3-Repeat $\times$ 5-Fold Cross-Validation, participant-level probability aggregation, and paired bootstrap statistical inference ($B = 2,000$, seed 42).

---

## 2. Main Experimental Findings

| Experimental Contrast | Baseline Model | Test Model | Metric Difference ($\Delta$) | 95% Bootstrap CI | Empirical Tail $p$-Value | Statistical Finding |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Morphometry Increment** | Clinical Only (AUROC: 0.8210) | Model A: Clin + FS29 (AUROC: 0.8643) | **+0.0433** | [+0.0135, +0.0747] | **0.0020** | **Statistically Significant ($p < 0.01$)** |
| **Stage 12: Whole-Volume 3D MRI** | Model A: Clin + FS29 (AUROC: 0.8517) | Model B: Clin + FS29 + 3D CNN (AUROC: 0.8587) | **+0.0070** | [-0.0057, +0.0197] | **0.2860** | **Not Significant ($p > 0.05$)** |
| **Stage 19: Subcortical ROI Contrast** | Model A: Clin + FS29 (AUROC: 0.8643) | Model B: Clin + FS29 + 16 ROIs (AUROC: 0.8688) | **+0.0044** | [-0.0029, +0.0122] | **0.2440** | **Not Significant ($p > 0.05$)** |

**Conclusion**: Regional FreeSurfer morphometry provides robust, statistically significant predictive signal beyond clinical testing. In contrast, adding raw 3D MRI volumes (via 3D CNN) or subcortical tissue contrast features does not yield a statistically significant performance improvement over a 24-month horizon.

---

## 3. Repository Architecture

```
MCI_AD_Research/
├── README.md                           # Master documentation
├── LICENSE                             # Academic research license
├── CITATION.cff                        # Machine-readable citation metadata
├── requirements.txt                    # Pinned Python package dependencies
├── pytest.ini                          # Testing configuration
│
├── docs/                               # Comprehensive project documentation
│   ├── project_overview.md             # Scientific context and summary
│   ├── research_objectives.md          # Primary research questions
│   ├── dataset_and_cohort.md           # Eligibility criteria and cohort flow
│   ├── methodology.md                  # Five-stage workflow architecture
│   ├── mri_preprocessing.md            # Image processing and QC gates
│   ├── feature_engineering.md          # Clinical, FreeSurfer, and ROI features
│   ├── model_architecture.md           # MLP, 3D CNN, and cross-attention
│   ├── experimental_design.md          # Multi-repeat CV and bootstrap protocol
│   ├── evaluation_metrics.md           # Metrics and benchmark results
│   ├── reproducibility.md              # Seeds, determinism, and software
│   ├── limitations.md                  # Scientific boundaries and caveats
│   ├── GITHUB_PUBLICATION_AUDIT.md     # Pre-publication forensic audit
│   └── reports/                        # Primary experiment reports
│
├── configs/                            # Configuration specifications
│   ├── paths.yaml                      # Centralized path resolver
│   ├── cohort.yaml                     # Inclusion/exclusion criteria
│   ├── preprocessing.yaml              # FSL and SimpleITK parameters
│   ├── features.yaml                   # Clinical, FS, and ROI definitions
│   ├── models/                         # Model hyperparameter configs
│   └── experiments/                    # Stage 12 and Stage 19 experiment configs
│
├── src/                                # Core modular Python package
│   ├── data/                           # Cohort filtering and split generators
│   ├── preprocessing/                  # MRI standardization pipeline
│   ├── features/                       # Clinical, FreeSurfer, and ROI extractors
│   ├── models/                         # MLPEnc, CNN3D, and MultimodalAttention
│   ├── training/                       # Cross-validation training routines
│   ├── evaluation/                     # Metric computers and paired bootstrap
│   └── utils/                          # Path helpers and environment utilities
│
├── scripts/                            # Standalone workflow drivers
│   ├── data_preparation/               # Cohort construction and split drivers
│   ├── preprocessing/                  # DICOM-to-NIfTI and registration
│   ├── feature_extraction/             # Feature table extraction drivers
│   ├── training/                       # Stage 12 and Stage 19 training scripts
│   ├── evaluation/                     # Paired bootstrap evaluation driver
│   └── quality_control/                # MRI QC audit scripts
│
├── results/                            # Curated aggregate outputs (No PII)
│   ├── summary_tables/                 # Benchmark CSV tables
│   ├── figures/                        # Forest plots and ROC curves
│   └── experiment_summaries/           # JSON configuration summaries
│
├── examples/                           # Interactive demonstration examples
│   ├── example_config.yaml             # Demo parameters
│   └── synthetic_data_example/         # Mock data generator and end-to-end demo
│
└── environment/                        # Reproducibility files
    ├── environment.yml                 # Conda environment definition
    └── software_versions.md            # Verified toolchain versions
```

---

## 4. Quick Start: Synthetic Demonstration

For researchers without immediate ADNI data access, run the complete pipeline on synthetic data:

```bash
# 1. Clone the repository
git clone https://github.com/Lahari468/MCI_AD_Research.git
cd MCI_AD_Research

# 2. Set up environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Run end-to-end synthetic pipeline
python examples/synthetic_data_example/run_synthetic_pipeline.py

# 4. Run automated test suite
pytest
```

---

## 5. Working with Authorized ADNI Data

1. **Obtain Data Approval**: Apply for research access at [ADNI LONI](https://adni.loni.usc.edu/).
2. **Download Baseline Data**: Download baseline clinical tables (`ADNIMERGE.csv`), FreeSurfer cross-sectional morphometry (`UCSFFSX7.csv`), and T1-weighted baseline DICOM collections.
3. **Configure Local Paths**: Update `configs/paths.yaml` with your local data directory paths.
4. **Build Cohort & Splits**:
   ```bash
   python scripts/data_preparation/build_cohort.py
   python scripts/data_preparation/generate_splits.py
   ```
5. **Preprocess MRI Volumes**:
   ```bash
   python scripts/preprocessing/preprocess_mri_volumes.py --roots <path_to_dicoms>
   ```
6. **Extract Subcortical Features**:
   ```bash
   python scripts/feature_extraction/extract_mri_subcortical_rois.py
   ```
7. **Execute Experiments**:
   ```bash
   # Run Stage 19 Subcortical ROI Incremental Experiment
   python scripts/training/run_stage19_mri_roi_incremental.py

   # Run Stage 12 Whole-Volume 3D MRI Ablation
   python scripts/training/run_stage12_3repeat_ablation.py
   ```

---

## 6. Privacy & ADNI Compliance Notice

This repository contains **strictly source code, configurations, documentation, and aggregate summary metrics**. It does not distribute any proprietary raw or processed participant data, individual scans, or participant identifiers. Researchers must comply with the ADNI Data Use Agreement.

---

## 7. License & Citation

- **License**: [Academic Research Code License](LICENSE)
- **Citation**: See [CITATION.cff](CITATION.cff)
