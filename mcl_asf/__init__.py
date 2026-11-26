"""Public MCL-ASF research implementation; no dataset or pretrained weights included."""
from .model import MCLASFPaperAligned, AdaptiveSpectralFusion, SpectralSpatialComplementaryEncoder, EXPECTED_BANDS, MODEL_INPUT_SIZE, CLASS_NAMES
from .losses import PaperSpecificHParams, cross_condition_mcl_loss, feature_consistency_loss, asf_weight_consistency_loss, paper_aligned_total_loss
from .augmentations import make_two_views, multi_perturbation_view
__all__ = ["MCLASFPaperAligned", "AdaptiveSpectralFusion", "SpectralSpatialComplementaryEncoder", "PaperSpecificHParams", "cross_condition_mcl_loss", "feature_consistency_loss", "asf_weight_consistency_loss", "paper_aligned_total_loss", "make_two_views", "multi_perturbation_view", "CLASS_NAMES"]
