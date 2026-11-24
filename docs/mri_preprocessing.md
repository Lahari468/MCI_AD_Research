# MRI Preprocessing Workflow

## Image Standardization Protocol
Raw T1-weighted baseline acquisitions from multiple scanner manufacturers (Siemens, GE, Philips) and field strengths (1.5T and 3.0T) were standardized through a deterministic pipeline:

```
Raw DICOM Series
    │
    ▼ (dcm2niix -z y -i y)
3D NIfTI Image
    │
    ▼ (fslreorient2std)
Standard Orientation (RPI)
    │
    ▼ (robustfov)
Neck & Non-Cranial Cropping
    │
    ▼ (SimpleITK N4ITK Bias Field Correction)
Corrected Intensity Homogeneity
    │
    ▼ (FSL FLIRT 12 DOF Affine Registration)
MNI152 Linear Standard Space (2.0 mm isotropic, 91 × 109 × 91 grid)
    │
    ▼ (Brain-Masked Robust Intensity Normalization)
Voxel Clipping [0.5%, 99.5%] & Z-Score Normalization
    │
    ▼
Export Tensor: (1, 91, 109, 91) float16
```

## Quality Control (QC) Gates
Each volume underwent automated quantitative quality auditing:
1. **Normalized Cross-Correlation (NCC)**: Correlation with the MNI152 template within intracranial mask ($NCC \ge 0.20$ required).
2. **Determinant Scale Factor**: Volume scaling constrained to $0.70 < \text{scale} < 1.40$.
3. **Rotation Angle**: Rigid rotational transformation limited to $\le 45^\circ$.
4. **Tri-Planar Slices**: Orthogonal view PNGs generated for visual alignment inspection.
