"""MCL-ASF: spectral-spatial complementary encoder and band-level ASF.

Adapted from the paper-aligned core; this release uses 1x1 spatial average pooling.
The class order is steel ball, tungsten bead, tungsten block, stone, background.
"""
from __future__ import annotations
from typing import Optional, Tuple, Dict
import torch
from torch import nn
import torch.nn.functional as F

EXPECTED_BANDS = 331
MODEL_INPUT_SIZE = 50
CLASS_NAMES = ("steel_ball", "tungsten_bead", "tungsten_block", "stone", "background")

class SpectralBandEncoder(nn.Module):
    """E_lambda: one d-dimensional representation per spectral band."""
    def __init__(self, feat_dim: int = 128):
        super().__init__()
        hidden = max(32, feat_dim // 2)
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, feat_dim, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
        )

    def forward(self, spectral_vector: torch.Tensor) -> torch.Tensor:
        # [N,B] -> [N,B,d]
        return self.net(spectral_vector.unsqueeze(1)).transpose(1, 2).contiguous()

class SpatialContextEncoder(nn.Module):
    """E_s: local spatial-context representation g_i in R^d."""
    def __init__(self, in_ch: int = EXPECTED_BANDS, feat_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(x).flatten(1))

class SpectralSpatialComplementaryEncoder(nn.Module):
    def __init__(self, bands: int = EXPECTED_BANDS, feat_dim: int = 128):
        super().__init__()
        self.bands = int(bands)
        self.feat_dim = int(feat_dim)
        self.spectral = SpectralBandEncoder(feat_dim=feat_dim)
        self.spatial = SpatialContextEncoder(in_ch=bands, feat_dim=feat_dim)
        self.gate = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spectral_vector = x.mean(dim=(2, 3))
        f_band = self.spectral(spectral_vector)       # Eq. (3): [N,B,d]
        g = self.spatial(x)                           # Eq. (3): [N,d]
        f_bar = f_band.mean(dim=1)
        gamma = self.gate(torch.cat([f_bar, g], dim=1))  # Eq. (4): [N,d]
        fused_band = (
            gamma.unsqueeze(1) * f_band
            + (1.0 - gamma).unsqueeze(1) * g.unsqueeze(1)
        )                                             # Eq. (5): [N,B,d]
        return fused_band, gamma, g

class AdaptiveSpectralFusion(nn.Module):
    """Eq. (6)--(8): softmax-normalized sample-specific band weighting."""
    def __init__(self, feat_dim: int = 128, energy_dim: Optional[int] = None):
        super().__init__()
        energy_dim = int(energy_dim or feat_dim)
        self.energy_fc = nn.Linear(feat_dim, energy_dim)
        self.energy_out = nn.Linear(energy_dim, 1, bias=False)

    def forward(self, fused_band: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        energy = self.energy_out(torch.tanh(self.energy_fc(fused_band))).squeeze(-1)  # [N,B]
        alpha = torch.softmax(energy, dim=1)                                           # Eq. (7)
        z = torch.sum(alpha.unsqueeze(-1) * fused_band, dim=1)                         # Eq. (8)
        return z, alpha, energy

class MCLASFPaperAligned(nn.Module):
    def __init__(self, num_classes: int = 5, bands: int = EXPECTED_BANDS, feat_dim: int = 128):
        super().__init__()
        self.num_classes = int(num_classes)
        self.bands = int(bands)
        self.feat_dim = int(feat_dim)
        self.encoder = SpectralSpatialComplementaryEncoder(bands=bands, feat_dim=feat_dim)
        self.asf = AdaptiveSpectralFusion(feat_dim=feat_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x_spatial: torch.Tensor, x_spectral: Optional[torch.Tensor] = None):
        # x_spectral is accepted only for backward compatibility with the old evaluator interface.
        fused_band, gamma, spatial_context = self.encoder(x_spatial)
        z, alpha, energy = self.asf(fused_band)
        logits = self.classifier(z)
        aux = {
            "gamma": gamma,
            "spatial_context": spatial_context,
            "band_features": fused_band,
            "energy": energy,
        }
        return logits, z, alpha, aux

