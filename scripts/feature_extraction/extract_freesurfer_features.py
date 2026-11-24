#!/usr/bin/env python3
"""
FreeSurfer Feature Processing Driver
Extracts the 29 FreeSurfer regional morphometry metrics normalized by intracranial volume.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.freesurfer import extract_freesurfer_features, save_freesurfer29_features

if __name__ == "__main__":
    print("Processing FreeSurfer FS29 feature table...")
    df, _ = extract_freesurfer_features()
    save_freesurfer29_features(df)
    print(f"FreeSurfer features successfully processed: shape {df.shape}")
