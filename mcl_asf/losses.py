"""Cross-condition metric contrastive learning and consistency objectives.

The mathematical operations are retained from the paper-aligned v2 core.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F

@dataclass(frozen=True)
class PaperSpecificHParams:
    """Six paper-specific hyperparameters that must be selected on indoor data only."""

    sigma_n: float
    max_shift: float
    band_dropout_p: float
    rho: float
    lambda_feat: float
    lambda_asf: float

    def validate(self) -> None:
        if self.sigma_n < 0:
            raise ValueError("sigma_n must be >= 0")
        if self.max_shift < 0:
            raise ValueError("max_shift must be >= 0")
        if not 0 <= self.band_dropout_p < 1:
            raise ValueError("band_dropout_p must be in [0, 1)")
        if self.rho < 0:
            raise ValueError("rho must be >= 0")
        if self.lambda_feat < 0 or self.lambda_asf < 0:
            raise ValueError("consistency weights must be >= 0")

    def to_dict(self) -> Dict[str, float]:
        self.validate()
        return {k: float(v) for k, v in asdict(self).items()}

def cross_condition_mcl_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    conditions: torch.Tensor,
    temperature: float = 0.08,
    rho: float = 1.0,
) -> torch.Tensor:
    """Weighted supervised contrastive objective corresponding to Eq. (10).

    omega_ip = 1 + rho * I(d_i != d_p) for same-class positive pairs.
    The denominator contains every non-self sample in the batch.
    """
    if features.ndim != 2:
        raise ValueError("features must be [N,d]")
    n = int(features.shape[0])
    if n <= 1:
        return features.sum() * 0.0

    z = F.normalize(features, dim=1)
    sim = torch.matmul(z, z.T) / float(temperature)
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()

    labels = labels.view(-1)
    conditions = conditions.view(-1)
    same_class = labels[:, None].eq(labels[None, :])
    not_self = ~torch.eye(n, dtype=torch.bool, device=features.device)
    positive = same_class & not_self

    exp_sim = torch.exp(sim) * not_self.to(sim.dtype)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    cross_condition = conditions[:, None].ne(conditions[None, :])
    weights = 1.0 + float(rho) * cross_condition.to(sim.dtype)
    weighted_positive = positive.to(sim.dtype) * weights

    pcount = positive.sum(dim=1)
    valid = pcount > 0
    if not torch.any(valid):
        return features.sum() * 0.0

    per_anchor = -(
        (weighted_positive * log_prob).sum(dim=1)[valid]
        / pcount[valid].to(sim.dtype)
    )
    return per_anchor.mean()

def feature_consistency_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    # Eq. (11), first term: mean squared L2 distance.
    return torch.mean(torch.sum((z1 - z2) ** 2, dim=1))

def asf_weight_consistency_loss(alpha1: torch.Tensor, alpha2: torch.Tensor) -> torch.Tensor:
    # Eq. (11), second term: KL(alpha^(1) || alpha^(2)).
    eps = 1e-8
    p = alpha1.clamp_min(eps)
    q = alpha2.clamp_min(eps)
    return torch.mean(torch.sum(p * (torch.log(p) - torch.log(q)), dim=1))

def paper_aligned_total_loss(
    logits1: torch.Tensor,
    logits2: torch.Tensor,
    z1: torch.Tensor,
    z2: torch.Tensor,
    alpha1: torch.Tensor,
    alpha2: torch.Tensor,
    labels: torch.Tensor,
    conditions: torch.Tensor,
    class_weights: Optional[torch.Tensor],
    hparams: PaperSpecificHParams,
    lambda_mcl: float = 0.25,
    temperature: float = 0.08,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    hparams.validate()
    labels2 = torch.cat([labels, labels], dim=0)
    conditions2 = torch.cat([conditions, conditions], dim=0)
    logits = torch.cat([logits1, logits2], dim=0)
    features = torch.cat([z1, z2], dim=0)

    l_cls = F.cross_entropy(logits, labels2, weight=class_weights)
    l_mcl = cross_condition_mcl_loss(
        features, labels2, conditions2,
        temperature=temperature, rho=hparams.rho,
    )
    l_feat = feature_consistency_loss(z1, z2)
    l_asf = asf_weight_consistency_loss(alpha1, alpha2)
    total = (
        l_cls
        + float(lambda_mcl) * l_mcl
        + float(hparams.lambda_feat) * l_feat
        + float(hparams.lambda_asf) * l_asf
    )
    return total, {
        "classification": l_cls,
        "mcl": l_mcl,
        "feature_consistency": l_feat,
        "asf_consistency": l_asf,
    }

