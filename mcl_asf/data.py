"""Dataset-independent patch manifest and preprocessing.

Expected manifest: patch_path,label,condition,source,cube_id,split.
Only training data marked split=train are accepted by the public training script.
"""
from __future__ import annotations
import csv
from pathlib import Path
from typing import Dict, Sequence, List, Any
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from .model import CLASS_NAMES, EXPECTED_BANDS, MODEL_INPUT_SIZE

CONDITION_NAMES = (
    "indoor_single_45klux",
    "indoor_single_65klux",
    "indoor_mix3_65klux",
)
SOURCES = ("indoor_single", "indoor_mix3")
CLASS_ALIASES = {
    "钢珠": 0, "钨珠": 1, "钨块": 2, "石子": 3, "白板": 4,
    "steel_ball": 0, "tungsten_bead": 1, "tungsten_block": 2, "stone": 3, "background": 4,
}


def parse_class(value: str) -> int:
    value = str(value).strip()
    if value in CLASS_ALIASES:
        return CLASS_ALIASES[value]
    try:
        out = int(value)
    except ValueError as exc:
        raise ValueError(f"Unknown class: {value!r}") from exc
    if out < 0 or out >= len(CLASS_NAMES):
        raise ValueError(f"Class must be 0..4, got {out}")
    return out


def normalize_patch_layout(patch: np.ndarray) -> np.ndarray:
    """Return float32 HWC array, preserving all 331 spectral bands."""
    patch = np.asarray(patch)
    if patch.ndim != 3:
        raise ValueError(f"Expected a three-dimensional patch, got {patch.shape}")
    if patch.shape[-1] == EXPECTED_BANDS:
        return np.asarray(patch, dtype=np.float32)
    if patch.shape[0] == EXPECTED_BANDS:
        return np.transpose(patch, (1, 2, 0)).astype(np.float32, copy=False)
    raise ValueError(f"Expected 331 bands in HWC or CHW order, got {patch.shape}")


def preprocess_patch(patch: np.ndarray, output_size: int = MODEL_INPUT_SIZE) -> torch.Tensor:
    """Match paper-aligned per-patch min-max normalization, then bilinear resize.

    Returns [331,output_size,output_size] float32. Accepts input [H,W,331] or [331,H,W].
    """
    x = normalize_patch_layout(patch)
    if not np.isfinite(x).all():
        raise ValueError("Non-finite values in patch")
    low, high = float(x.min()), float(x.max())
    x = (x - low) / (high - low + 1e-6)
    chw = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
    if tuple(chw.shape[1:]) != (output_size, output_size):
        chw = F.interpolate(chw.unsqueeze(0), size=(output_size, output_size),
                            mode="bilinear", align_corners=False).squeeze(0)
    return chw


def load_manifest(manifest, check_files: bool = True) -> List[Dict[str, Any]]:
    path = Path(manifest).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    required = {"patch_path", "label", "condition", "source", "cube_id", "split"}
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")
        for i, row in enumerate(reader, 2):
            if row["split"].strip().lower() != "train":
                raise ValueError(f"Training manifest must contain train rows only, line {i}: {row['split']}")
            cond = row["condition"].strip()
            source = row["source"].strip()
            if cond not in CONDITION_NAMES:
                raise ValueError(f"Unknown condition on line {i}: {cond}")
            if source not in SOURCES:
                raise ValueError(f"Unknown source on line {i}: {source}")
            if cond.startswith("indoor_mix3") and source != "indoor_mix3":
                raise ValueError(f"Condition/source mismatch on line {i}")
            if cond.startswith("indoor_single") and source != "indoor_single":
                raise ValueError(f"Condition/source mismatch on line {i}")
            candidate = Path(row["patch_path"].strip())
            if not candidate.is_absolute():
                candidate = path.parent / candidate
            if check_files and not candidate.is_file():
                raise FileNotFoundError(f"Missing patch on line {i}: {candidate}")
            if not str(row["cube_id"]).strip():
                raise ValueError(f"Missing cube_id on line {i}")
            rows.append({
                "patch_path": candidate,
                "label": parse_class(row["label"]),
                "condition": CONDITION_NAMES.index(cond),
                "condition_name": cond,
                "source": source,
                "cube_id": str(row["cube_id"]).strip(),
            })
    if not rows:
        raise ValueError("Empty training manifest")
    return rows


class PatchManifestDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, Any]], output_size: int = MODEL_INPUT_SIZE):
        self.rows = list(rows)
        self.output_size = int(output_size)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        x = preprocess_patch(np.load(row["patch_path"], allow_pickle=False), self.output_size)
        return x, torch.tensor(row["label"], dtype=torch.long), torch.tensor(row["condition"], dtype=torch.long)
