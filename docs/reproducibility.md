# Reproducibility Guide

## Software Environment
- **Python**: 3.9.6
- **Key Libraries**: PyTorch 2.8.0, scikit-learn 1.6.1, NumPy 2.0.2, Pandas 2.3.3, NiBabel 5.3.3, SimpleITK 2.5.7.
- Complete pinned environment details are documented in `environment/software_versions.md` and `environment/environment.yml`.

## Determinism & Seeds
- Global random seed: `42`.
- Fold-specific seeds: `42 + 100 * repeat + fold`.
- Resampling seed for paired bootstrap: `42`.
- Execution on CPU guarantees deterministic floating-point operations across runs.

## End-to-End Synthetic Demonstration
For users without authorized ADNI data access, a synthetic demonstration pipeline is provided:
```bash
python examples/synthetic_data_example/run_synthetic_pipeline.py
```
This generates non-identifiable mock tabular and volumetric data, executes the baseline and cross-attention architectures, computes evaluation metrics, and performs a paired bootstrap comparison out-of-the-box.
