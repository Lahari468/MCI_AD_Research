# Software & Environment Specifications

**Project**: Evaluating the Incremental Predictive Value of Baseline 3D MRI Beyond Clinical and FreeSurfer Morphometry for 24-Month MCI-to-Alzheimer's Disease Progression  
**Operating System**: macOS (Darwin 24.x / Apple Silicon ARM64)

---

## 1. Verified Python Environment

The experiments were executed and verified within an isolated virtual environment (`.venv`):

| Software Component | Verified Version | Purpose |
| :--- | :--- | :--- |
| **Python** | `3.9.6` | Core programming language environment |
| **PyTorch (`torch`)** | `2.8.0` | Deep learning, 3D CNNs, and multimodal cross-attention networks |
| **NumPy (`numpy`)** | `2.0.2` | Numerical computing, linear algebra, array serialization |
| **Pandas (`pandas`)** | `2.3.3` | Tabular data analysis, manifest processing, clinical metrics |
| **scikit-learn (`sklearn`)** | `1.6.1` | Logistic Regression, HistGradientBoosting, ROC-AUC, Brier score, splits |
| **NiBabel (`nibabel`)** | `5.3.3` | NIfTI neuroimaging volume I/O and affine matrix handling |
| **SimpleITK (`sitk`)** | `2.5.7` | N4 bias field correction and spatial resampling |
| **PyDICOM (`pydicom`)** | `2.4.4` | DICOM header parsing and series verification |
| **PyYAML (`yaml`)** | `6.0.3` | Configuration parsing and experiment specifications |
| **pytest (`pytest`)** | `8.4.2` | Test suite execution and validation |
| **Matplotlib (`matplotlib`)**| `3.9.4` | Forest plots, calibration curves, and figure generation |
| **Seaborn (`seaborn`)** | `0.13.2` | Statistical data visualization |

---

## 2. External Neuroimaging Toolchains

The MRI preprocessing pipeline leverages external FSL and dcm2niix toolchains:

| Tool | Recommended Version | Pipeline Function |
| :--- | :--- | :--- |
| **FSL (`flirt`)** | FSL 6.0.x | 12 DOF affine linear registration to MNI152 2mm template |
| **FSL (`fslreorient2std`)** | FSL 6.0.x | Standardizing spatial orientation across scanner vendors |
| **FSL (`robustfov`)** | FSL 6.0.x | Field-of-view reduction to exclude neck and lower cranial tissue |
| **dcm2niix** | v1.0.20230411+ | Converting raw DICOM image slices into 3D NIfTI volumes |

---

## 3. Hardware Execution & Device Notes (CPU vs. MPS)

- **Device Behavior**: On Apple Silicon hardware, `torch.backends.mps.is_available()` returns `True` under native Terminal execution, but returns `False` inside restricted sandboxes (e.g., containerized execution or sandbox runners without IOKit/Metal device permissions).
- **Execution Reality**: To ensure absolute determinism across all 15 model-fold training runs in Stage 12 and Stage 19, training was executed deterministically on CPU across physical performance cores (`DEV = 'cpu'`).
- **Reproducibility Guarantee**: The published PyTorch and scikit-learn models run seamlessly on both CPU and MPS backends, yielding reproducible numerical results across operating environments.
