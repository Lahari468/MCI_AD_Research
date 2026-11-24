#!/usr/bin/env python3
"""
Clinical Feature Processing Driver
Extracts the 10 core clinical assessment scores from raw ADNI tabular data.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.clinical import extract_clinical_features, save_clinical_features

if __name__ == "__main__":
    print("Processing clinical feature table...")
    df = extract_clinical_features()
    save_clinical_features(df)
    print(f"Clinical features successfully processed: shape {df.shape}")
