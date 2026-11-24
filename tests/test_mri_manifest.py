"""
Automated Validation Suite for Master MRI Manifest
File: tests/test_mri_manifest.py
"""

import os, hashlib, random
import numpy as np
import pandas as pd
import pytest
from src.utils.paths import get_project_root, load_manifest, get_dicom_dir, get_tensor_path

def test_manifest_dimensions_and_keys():
    mf = load_manifest()
    assert len(mf) == 713, f"Expected 713 rows, got {len(mf)}"
    assert mf["rid"].nunique() == 713, "Duplicate RIDs detected in manifest!"
    assert mf["image_id"].nunique() == 713, "Duplicate image_ids detected in manifest!"

def test_origin_counts():
    mf = load_manifest()
    counts = mf["origin"].value_counts().to_dict()
    assert counts.get("original520") == 520, f"Expected 520 original, got {counts.get('original520')}"
    assert counts.get("recovered193") == 193, f"Expected 193 recovered, got {counts.get('recovered193')}"

def test_source_paths_and_file_counts():
    mf = load_manifest()
    root = get_project_root()
    for idx, r in mf.iterrows():
        p = root / str(r["source_path_current"])
        assert p.exists(), f"Source path does not exist: {p}"
        files = [f for f in os.listdir(p) if not f.startswith(".")]
        assert len(files) == int(r["dicom_files"]), f"File count mismatch for image {r['image_id']}: {len(files)} vs {r['dicom_files']}"

def test_qc_status_counts():
    mf = load_manifest()
    counts = mf["qc_status"].value_counts().to_dict()
    assert counts.get("OK") == 656, f"Expected 656 OK, got {counts.get('OK')}"
    assert counts.get("REVIEW") == 48, f"Expected 48 REVIEW, got {counts.get('REVIEW')}"
    assert counts.get("FAIL") == 8, f"Expected 8 FAIL, got {counts.get('FAIL')}"
    assert counts.get("ERROR") == 1, f"Expected 1 ERROR, got {counts.get('ERROR')}"

def test_tensors_shape_finiteness_and_hashes():
    mf = load_manifest()
    root = get_project_root()
    
    # 712 tensors present, RID 2373 absent
    tensors_present = mf["tensor_present"].sum()
    assert tensors_present == 712, f"Expected 712 tensors present, got {tensors_present}"
    
    r2373 = mf[mf["rid"] == 2373].iloc[0]
    assert not r2373["tensor_present"], "RID 2373 tensor must be absent!"
    assert not r2373["mri_usable"], "RID 2373 mri_usable must be False!"
    
    # Check 20 sampled tensors with fixed seed
    random.seed(42)
    sample_rids = random.sample(mf[mf["tensor_present"]]["rid"].tolist(), 20)
    for rid in sample_rids:
        row = mf[mf["rid"] == rid].iloc[0]
        tp = root / str(row["tensor_path"])
        assert tp.exists(), f"Tensor {tp} missing!"
        arr = np.load(tp)
        assert arr.shape == (1, 91, 109, 91), f"Wrong shape {arr.shape} for RID {rid}"
        assert arr.dtype == np.float16, f"Wrong dtype {arr.dtype} for RID {rid}"
        assert np.all(np.isfinite(arr)), f"Non-finite values detected in RID {rid}"
        # Check recomputed sha256 equals manifest
        recomp_sha = hashlib.sha256(open(tp, "rb").read()).hexdigest()
        assert recomp_sha == row["tensor_sha256"], f"SHA256 mismatch for RID {rid}"

def test_field_strength_bins():
    mf = load_manifest()
    assert set(mf["field_strength_normalized"].unique()) == {1.5, 3.0}, "Normalized field strength must be in {1.5, 3.0}!"
