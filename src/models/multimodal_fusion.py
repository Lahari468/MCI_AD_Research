"""
Multimodal Cross-Attention Fusion Network
Module: src/models/multimodal_fusion.py
Implements Stage 12 Model A (MRI excluded) and Model B (MRI included).
"""

import torch
import torch.nn as nn
from src.models.encoders import MLPEnc
from src.models.cnn3d import CNN3D


class MultimodalAttentionNetwork(nn.Module):
    """
    Multimodal Compact Cross-Attention Network with Missingness Conditioning.
    Fuses Clinical (10), FreeSurfer (29), and 3D MRI (volumetric) modalities.
    """
    def __init__(
        self,
        d_clinical: int = 10,
        d_freesurfer: int = 29,
        latent_dim: int = 32,
        num_heads: int = 2,
        mri_channels: int = 1,
        use_mri: bool = False,
        use_fs: bool = True
    ):
        super().__init__()
        self.use_mri = use_mri
        self.use_fs = use_fs
        self.latent_dim = latent_dim

        # Modality Encoders
        self.enc_clinical = MLPEnc(d_clinical, latent_dim)
        self.enc_freesurfer = MLPEnc(d_freesurfer, latent_dim) if use_fs else None
        self.enc_mri = CNN3D(mri_channels, latent_dim) if use_mri else None

        # Modality Type Embeddings (3 modalities: Clinical, FS, MRI)
        self.modality_embeddings = nn.Parameter(torch.randn(3, latent_dim) * 0.02)

        # Cross-Modal Multi-Head Self-Attention
        self.attn = nn.MultiheadAttention(latent_dim, num_heads, dropout=0.1, batch_first=True)
        self.ln1 = nn.LayerNorm(latent_dim)
        self.ln2 = nn.LayerNorm(latent_dim)
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, 2 * latent_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2 * latent_dim, latent_dim)
        )

        # Fusion Classification Head conditioned on Modality Presence Mask
        h_dim = 2 * latent_dim
        self.fuse = nn.Sequential(
            nn.Linear(3 * latent_dim + 3, h_dim),
            nn.LayerNorm(h_dim),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(h_dim, h_dim // 2),
            nn.LayerNorm(h_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        self.head = nn.Linear(h_dim // 2, 1)

    def forward(
        self,
        x_clinical: torch.Tensor,
        x_freesurfer: torch.Tensor = None,
        x_mri: torch.Tensor = None,
        modality_mask: torch.Tensor = None
    ):
        """
        x_clinical: (B, 10)
        x_freesurfer: (B, 29)
        x_mri: (B, 1, D, H, W) or None
        modality_mask: (B, 3) binary presence indicators [clin, fs, mri]
        """
        B = x_clinical.size(0)
        if modality_mask is None:
            modality_mask = torch.ones(B, 3, device=x_clinical.device)

        ec = self.enc_clinical(x_clinical)

        if self.enc_freesurfer is not None and x_freesurfer is not None:
            ef = self.enc_freesurfer(x_freesurfer)
        else:
            ef = torch.zeros_like(ec)

        em = torch.zeros_like(ec)
        if self.enc_mri is not None and x_mri is not None:
            mask_mri_bool = (modality_mask[:, 2] > 0.5)
            if mask_mri_bool.any():
                em[mask_mri_bool] = self.enc_mri(x_mri)

        # Stack into token sequence (B, 3, latent_dim) with modality embeddings
        tokens = (torch.stack([ec, ef, em], dim=1) + self.modality_embeddings) * modality_mask.unsqueeze(-1)

        # Multi-Head Attention with key padding mask (mask out missing modalities)
        key_padding = (modality_mask < 0.5)
        att, weights = self.attn(tokens, tokens, tokens, key_padding_mask=key_padding, need_weights=True)
        tokens = self.ln1(tokens + att)
        tokens = self.ln2(tokens + self.ffn(tokens)) * modality_mask.unsqueeze(-1)

        # Concatenate tokens with modality mask and project to logit
        z = self.fuse(torch.cat([tokens.flatten(1), modality_mask], dim=1))
        logit = self.head(z).squeeze(-1)
        return logit, weights
