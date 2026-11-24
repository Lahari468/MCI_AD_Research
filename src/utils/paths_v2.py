"""
Centralized Path and Manifest Resolver
Module: src/utils/paths.py (and paths_v2.py)
"""

import os
from pathlib import Path
import pandas as pd

def get_project_root() -> Path:
    """Return project root Path from environment or default to current directory parent."""
    env_root = os.environ.get("PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()
    # If in standard repo location
    p = Path(__file__).resolve().parent.parent.parent
    if (p / "data").exists() and (p / "configs").exists():
        return p
    return Path(".").resolve()

def get_fsl_dir() -> Path:
    """Return FSLDIR from environment or default host installation."""
    fsl = os.environ.get("FSLDIR", "/usr/local/fsl")
    return Path(fsl).resolve()

_MANIFEST_CACHE = None

def load_manifest() -> pd.DataFrame:
    """Load and cache the master 713 MRI manifest."""
    global _MANIFEST_CACHE
    if _MANIFEST_CACHE is None:
        p = get_project_root() / "data/processed/mri/manifests/mri_manifest_713.csv"
        if not p.exists():
            p = get_project_root() / "results/audit/mri_scan_table_713.csv"
        _MANIFEST_CACHE = pd.read_csv(p)
    return _MANIFEST_CACHE.copy()

def get_dicom_dir(image_id: int) -> Path:
    """Return the absolute Path to the DICOM series folder for a given image_id."""
    mf = load_manifest()
    m = mf[mf["image_id"] == int(image_id)]
    if len(m) == 0:
        raise KeyError(f"Image ID {image_id} not found in master manifest.")
    rel_path = m["source_path_current"].values[0]
    return get_project_root() / rel_path

def get_tensor_path(rid: int) -> Path:
    """Return the Path to the processed tensor (.npy) for a given RID."""
    mf = load_manifest()
    m = mf[mf["rid"] == int(rid)]
    if len(m) == 0:
        raise KeyError(f"RID {rid} not found in master manifest.")
    tensor_path = m["tensor_path"].values[0]
    if pd.isna(tensor_path) or not tensor_path:
        raise FileNotFoundError(f"No tensor path registered for RID {rid} (e.g. localizer / excluded scan).")
    return get_project_root() / tensor_path
