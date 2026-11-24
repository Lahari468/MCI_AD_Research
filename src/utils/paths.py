"""
Centralized Path and Manifest Resolver
Module: src/utils/paths.py
Provides portable, environment-independent paths for project assets.
"""

import os
from pathlib import Path
import pandas as pd


def get_project_root() -> Path:
    """Return project root Path from environment or resolve from repository structure."""
    env_root = os.environ.get("PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()
    # Resolve relative to this file's position: src/utils/paths.py -> repo root
    p = Path(__file__).resolve().parent.parent.parent
    if (p / "configs").exists():
        return p
    return Path(".").resolve()


def get_fsl_dir() -> Path:
    """Return FSLDIR from environment."""
    fsl = os.environ.get("FSLDIR", "")
    return Path(fsl).resolve() if fsl else Path("/usr/local/fsl")


_MANIFEST_CACHE = None


def load_manifest(manifest_path: str = None) -> pd.DataFrame:
    """Load and cache the MRI manifest."""
    global _MANIFEST_CACHE
    if _MANIFEST_CACHE is None:
        if manifest_path is None:
            p = get_project_root() / "data/processed/mri/manifests/mri_manifest_713.csv"
            if not p.exists():
                p = get_project_root() / "results/audit/mri_scan_table_713.csv"
        else:
            p = Path(manifest_path)
        if p.exists():
            _MANIFEST_CACHE = pd.read_csv(p)
        else:
            _MANIFEST_CACHE = pd.DataFrame()
    return _MANIFEST_CACHE.copy()


def get_dicom_dir(image_id: int) -> Path:
    """Return the Path to the DICOM series folder for a given image_id."""
    mf = load_manifest()
    if mf.empty:
        raise FileNotFoundError("MRI manifest not found. Please configure authorized data paths.")
    m = mf[mf["image_id"] == int(image_id)]
    if len(m) == 0:
        raise KeyError(f"Image ID {image_id} not found in manifest.")
    rel_path = m["source_path_current"].values[0]
    return get_project_root() / rel_path


def get_tensor_path(rid: int) -> Path:
    """Return the Path to the processed tensor (.npy) for a given RID."""
    mf = load_manifest()
    if mf.empty:
        raise FileNotFoundError("MRI manifest not found. Please configure authorized data paths.")
    m = mf[mf["rid"] == int(rid)]
    if len(m) == 0:
        raise KeyError(f"RID {rid} not found in manifest.")
    tensor_path = m["tensor_path"].values[0]
    if pd.isna(tensor_path) or not tensor_path:
        raise FileNotFoundError(f"No tensor path registered for RID {rid}.")
    return get_project_root() / tensor_path
