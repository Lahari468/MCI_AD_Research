#!/usr/bin/env python3
"""
Synthetic Data Generator
Generates non-identifiable mock clinical, FreeSurfer, and 3D MRI tensors for pipeline demonstration
without exposing any real ADNI participant data.
"""

import os
import numpy as np
import pandas as pd


def generate_synthetic_dataset(output_dir: str = "examples/synthetic_data_example/data", n_samples: int = 40):
    os.makedirs(f"{output_dir}/mri_tensors", exist_ok=True)
    rng = np.random.default_rng(42)

    # 1. Generate Synthetic Labels (20% positive conversion)
    y = (rng.uniform(0, 1, n_samples) < 0.20).astype(int)
    subject_ids = [f"SYN_{i:04d}" for i in range(1, n_samples + 1)]

    # 2. Generate Synthetic Clinical Assessments (10 features)
    c_cols = ['age', 'sex', 'PTEDUCAT', 'APOE4', 'MMSCORE', 'CDRSB', 'CDGLOBAL', 'FAQTOTAL', 'TOTSCORE', 'TOTAL13']
    clin_data = {
        'subject_id': subject_ids,
        'label': y,
        'age': rng.normal(73.5, 7.0, n_samples).round(1),
        'sex': rng.choice([0, 1], n_samples),
        'PTEDUCAT': rng.integers(10, 21, n_samples),
        'APOE4': rng.choice([0, 1, 2], n_samples, p=[0.55, 0.35, 0.10]),
        'MMSCORE': rng.integers(24, 31, n_samples),
        'CDRSB': rng.choice([0.5, 1.0, 1.5, 2.0, 2.5], n_samples),
        'CDGLOBAL': np.full(n_samples, 0.5),
        'FAQTOTAL': rng.integers(0, 15, n_samples),
        'TOTSCORE': rng.normal(12.0, 5.0, n_samples).round(1),
        'TOTAL13': rng.normal(18.0, 7.0, n_samples).round(1)
    }
    df_clin = pd.DataFrame(clin_data)
    df_clin.to_csv(f"{output_dir}/synthetic_clinical.csv", index=False)

    # 3. Generate Synthetic FreeSurfer Morphometry (29 regional volumes/thicknesses)
    fs_cols = [f"ST{i:02d}SV" for i in range(1, 25)] + [f"ST{i:02d}TA" for i in range(25, 30)]
    fs_data = {'subject_id': subject_ids, 'ST10CV': rng.normal(1500000, 120000, n_samples).round(0)}
    for col in fs_cols:
        if col.endswith('SV'):
            fs_data[col] = rng.normal(3500, 600, n_samples).round(1)
        else:
            fs_data[col] = rng.normal(2.5, 0.3, n_samples).round(3)
    df_fs = pd.DataFrame(fs_data)
    df_fs.to_csv(f"{output_dir}/synthetic_freesurfer.csv", index=False)

    # 4. Generate Synthetic 3D MRI Tensors (Small downsampled 32x32x32 volumes)
    for sid in subject_ids:
        # Create a mock brain tensor with a synthetic sphere
        vol = rng.normal(0.0, 0.1, (1, 32, 32, 32)).astype(np.float32)
        np.save(f"{output_dir}/mri_tensors/{sid}.npy", vol)

    print(f"Synthetic dataset generated successfully in: {output_dir}")
    print(f"Total Participants: {n_samples} (Positives: {y.sum()}, Negatives: {n_samples - y.sum()})")


if __name__ == "__main__":
    generate_synthetic_dataset()
