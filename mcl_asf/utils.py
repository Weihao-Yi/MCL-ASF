"""Reproducibility and checkpoint helpers."""
from __future__ import annotations
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
from .model import MCLASFPaperAligned


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (TypeError, AttributeError):
        pass


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def load_checkpoint(path: str, device: torch.device):
    """Load only trusted checkpoints; training code saves plain tensor state dicts."""
    try:
        obj = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location=device)
    if not isinstance(obj, dict):
        raise TypeError("Expected a PyTorch dictionary checkpoint")
    state = obj.get("state_dict", obj)
    config = obj.get("model_config", {"bands":331,"num_classes":5,"feat_dim":128})
    model = MCLASFPaperAligned(**config)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), obj
