"""Independent composite perturbation views for MCL-ASF."""
from __future__ import annotations
from typing import Tuple
import torch
from .losses import PaperSpecificHParams

def gaussian_noise(x: torch.Tensor, sigma_n: float) -> torch.Tensor:
    if sigma_n <= 0:
        return x
    return x + torch.randn_like(x) * float(sigma_n)

def spectral_axis_shift(x: torch.Tensor, max_shift: float) -> torch.Tensor:
    """Per-sample fractional shift along the B axis with 1-D linear interpolation.

    x: [N, B, H, W]. A positive delta samples x(b + delta), using linear interpolation.
    """
    if max_shift <= 0:
        return x
    if x.ndim != 4:
        raise ValueError(f"spectral_axis_shift expects [N,B,H,W], got {tuple(x.shape)}")
    n, b, h, w = x.shape
    delta = (torch.rand(n, device=x.device, dtype=x.dtype) * 2.0 - 1.0) * float(max_shift)
    base = torch.arange(b, device=x.device, dtype=x.dtype).view(1, b)
    pos = torch.clamp(base + delta.view(n, 1), 0.0, float(b - 1))
    lo = torch.floor(pos).long()
    hi = torch.clamp(lo + 1, max=b - 1)
    frac = (pos - lo.to(pos.dtype)).view(n, b, 1, 1)
    lo4 = lo.view(n, b, 1, 1).expand(n, b, h, w)
    hi4 = hi.view(n, b, 1, 1).expand(n, b, h, w)
    x_lo = torch.gather(x, dim=1, index=lo4)
    x_hi = torch.gather(x, dim=1, index=hi4)
    return x_lo * (1.0 - frac) + x_hi * frac

def random_band_dropout(x: torch.Tensor, p_b: float) -> torch.Tensor:
    if p_b <= 0:
        return x
    if not 0 <= p_b < 1:
        raise ValueError("p_b must be in [0,1)")
    n, b, _, _ = x.shape
    mask = torch.bernoulli(
        torch.full((n, b, 1, 1), 1.0 - float(p_b), device=x.device, dtype=x.dtype)
    )
    return x * mask

def multi_perturbation_view(
    x: torch.Tensor,
    sigma_n: float,
    max_shift: float,
    band_dropout_p: float,
    clamp: bool = True,
) -> torch.Tensor:
    # Eq. (2): T_comp = T_drop o T_shift o T_noise
    y = gaussian_noise(x, sigma_n)
    y = spectral_axis_shift(y, max_shift)
    y = random_band_dropout(y, band_dropout_p)
    return y.clamp(0.0, 1.0) if clamp else y

def make_two_views(x: torch.Tensor, hparams: PaperSpecificHParams) -> Tuple[torch.Tensor, torch.Tensor]:
    hparams.validate()
    v1 = multi_perturbation_view(x, hparams.sigma_n, hparams.max_shift, hparams.band_dropout_p)
    v2 = multi_perturbation_view(x, hparams.sigma_n, hparams.max_shift, hparams.band_dropout_p)
    return v1, v2

