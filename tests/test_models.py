"""
Unit tests for model architectures and baseline classifiers.
"""

import torch
import numpy as np
from src.models.encoders import MLPEnc
from src.models.cnn3d import CNN3D
from src.models.multimodal_fusion import MultimodalAttentionNetwork
from src.models.baselines import LogisticRegressionBaseline, select_threshold_youden


def test_mlp_encoder():
    enc = MLPEnc(d_in=10, d_out=32)
    x = torch.randn(4, 10)
    out = enc(x)
    assert out.shape == (4, 32), f"Expected shape (4, 32), got {out.shape}"


def test_cnn3d():
    cnn = CNN3D(c_in=1, d_out=32, channel_widths=(8, 16, 24, 32))
    # Small synthetic volume: (batch, 1, 16, 16, 16)
    vol = torch.randn(2, 1, 16, 16, 16)
    out = cnn(vol)
    assert out.shape == (2, 32), f"Expected shape (2, 32), got {out.shape}"


def test_multimodal_fusion_forward():
    model_a = MultimodalAttentionNetwork(d_clinical=10, d_freesurfer=29, latent_dim=32, use_mri=False)
    xc = torch.randn(3, 10)
    xf = torch.randn(3, 29)
    mask = torch.ones(3, 3)

    logit_a, weights_a = model_a(xc, xf, None, mask)
    assert logit_a.shape == (3,), f"Expected logit shape (3,), got {logit_a.shape}"
    assert weights_a.shape == (3, 3, 3), f"Expected attention weights shape (3, 3, 3), got {weights_a.shape}"


def test_logistic_regression_baseline():
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (50, 10))
    y = rng.choice([0, 1], size=50)

    clf = LogisticRegressionBaseline(random_state=42)
    clf.fit(X[:35], y[:35], X[35:], y[35:])
    probs = clf.predict_proba(X[35:])
    preds = clf.predict(X[35:])

    assert len(probs) == 15
    assert len(preds) == 15
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
