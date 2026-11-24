"""
3D Convolutional Neural Network for Volumetric Brain MRI
Module: src/models/cnn3d.py
"""

import torch
import torch.nn as nn


class CNN3D(nn.Module):
    """
    4-Stage 3D CNN with Group Normalization and Dual Spatial Pooling (Mean + Max).
    Extracts deep anatomical representations from 3D brain MRI volumes.
    """
    def __init__(self, c_in: int = 1, d_out: int = 32, channel_widths=(16, 32, 64, 96)):
        super().__init__()
        layers = []
        c = c_in
        for o in channel_widths:
            layers += [
                nn.Conv3d(c, o, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(num_groups=4, num_channels=o),
                nn.GELU(),
                nn.Conv3d(o, o, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups=4, num_channels=o),
                nn.GELU()
            ]
            c = o
        self.features = nn.Sequential(*layers)
        self.out = nn.Sequential(
            nn.Linear(2 * c, d_out),
            nn.LayerNorm(d_out),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, 1, D, H, W)
        returns: (batch_size, d_out)
        """
        h = self.features(x)
        # Dual spatial pooling across D, H, W (dims 2, 3, 4)
        pooled = torch.cat([h.mean(dim=(2, 3, 4)), h.amax(dim=(2, 3, 4))], dim=1)
        return self.out(pooled)
