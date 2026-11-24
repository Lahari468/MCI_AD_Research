"""
Tabular Feature Encoders for Clinical and FreeSurfer Morphometry
Module: src/models/encoders.py
"""

import torch
import torch.nn as nn


class MLPEnc(nn.Module):
    """
    3-Layer Multi-Layer Perceptron Encoder with LayerNorm, GELU, and Dropout.
    Maps 1D tabular feature vectors (Clinical or FreeSurfer) to a shared latent dimension.
    """
    def __init__(self, d_in: int, d_out: int = 32, dropouts=(0.3, 0.2, 0.1)):
        super().__init__()
        dims = [d_in, 128, 256, d_out]
        layers = []
        for i in range(3):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]),
                nn.GELU(),
                nn.Dropout(dropouts[i])
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
