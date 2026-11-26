# MCL-ASF

**MCL-ASF: Ground-based Hyperspectral Small-Target Detection via Multi-Perturbation Spectral-Spatial Complementary Representation**

A public, dataset-independent implementation of the model and losses accompanying the manuscript. This repository contains source code only. **The hyperspectral dataset and trained model checkpoints are not publicly distributed.**

## Architecture

1. **MSCR (Multi-Perturbation Spectral–Spatial Complementary Representation)**: Gaussian noise, fractional spectral shifts and band dropout.
2. MSCR complementary encoder: spectral band encoding and spatial-context encoding with sample-dependent gating.
3. **ASF**: per-band softmax weights, weighted aggregation of band-level features.
4. **MCL**: supervised cross-condition contrastive loss plus feature and ASF-weight consistency regularization.
5. Five-class classifier: steel ball, tungsten bead, tungsten block, stone, background. The first four Softmax scores are used for separate one-vs-rest detection tasks.

```
Input patches [N,331,50,50]
  ├── Multi-perturbation views (training only)
  └── MSCR spectral + spatial encoding (shared encoder)
        └── ASF band weighting → representation
              ├── 5-class logits → 4 one-vs-rest scores
              └── MCL + feature / ASF consistency (training only)
```

## Repository structure

```text
mcl_asf/
  model.py         # MSCR + ASF + classification
  losses.py        # MCL + feature/ASF consistency + CE objective
  augmentations.py # composite spectral views
  data.py          # manifest-based private data adapter
  utils.py         # seed and safe checkpoint loading
scripts/
  train.py         # indoor patch training
  infer_patches.py # inference on .npy patches
  infer_cube.py    # optional full cube score maps
  smoke_test.py    # random-data structural test
configs/
  train.example.json
examples/
  train_manifest.example.csv
tests/
  test_core.py
```

## Installation

Python 3.8 or newer, PyTorch and NumPy:

```bash
python -m pip install -e .
python -m scripts.smoke_test
python -m unittest discover -s tests -v
```

A smoke-test pass establishes basic forward and backward execution **without dataset access**. It does not validate paper detection performance.

## Private data interface

Training accepts a user-provided CSV file with columns:

| Field | Meaning |
|---|---|
| `patch_path` | Absolute path or path relative to CSV location of an `.npy` patch; shape H×W×331 or 331×H×W |
| `label` | Integer 0..4 (steel ball, tungsten bead, tungsten block, stone, background), or class name |
| `condition` | `indoor_single_45klux`, `indoor_single_65klux`, `indoor_mix3_65klux` |
| `source` | `indoor_single` or `indoor_mix3` |
| `cube_id` | Identity of original source cube for group-aware splitting and auditing |
| `split` | `train`; leave held-out test cubes out of the training manifest |

See `examples/train_manifest.example.csv`; its paths are **illustrative and intentionally do not exist**.

Training with existing private indoor patches:

```bash
python -m scripts.train --manifest /path/to/private_train_manifest.csv --config configs/train.example.json --preflight-only
python -m scripts.train --manifest /path/to/private_train_manifest.csv --config configs/train.example.json --out outputs/train --seeds 0-9
```

The example configuration checks for **80 single-target and 20 mixed-target indoor cubes**, matching the later dataset description. For a different public dataset, pass `--allow-cube-count-mismatch` and define the appropriate condition/source mapping. No ground-based test cube is needed for training.

The example configuration uses the revised manuscript settings: Gaussian noise standard deviation 0.01, maximum spectral shift 1 band, and band-dropout probability 0.05.

## Inference

For individual patches:

```bash
python -m scripts.infer_patches --checkpoint outputs/train/mcl_asf_seed_00.pth \
    --patches /path/to/patch1.npy /path/to/patch2.npy \
    --output outputs/predictions.csv
```

Optional full-scene `.npy` cube with H×W×331 layout:

```bash
python -m scripts.infer_cube --checkpoint outputs/train/mcl_asf_seed_00.pth \
    --cube /path/to/cube.npy --stride 20 --patch-size 20 \
    --output outputs/score4_raw.npy
```

Output `score4_raw.npy` has shape `[4,H,W]` in the four target class order. It contains bilinearly interpolated Softmax probabilities and does **not** use Ground Truth to enhance images. For large scenes, dense inference at stride 2 can be very time-consuming on CPU.

## Reproducibility and data availability

The provided data adapter can be connected to external/private HSI datasets. The original sensor-specific ENVI XML ROI reader, full dataset and pretrained models are deliberately excluded. Reproducing manuscript tables requires the dataset, trained checkpoints, evaluation split and full detection protocol.

**Model size:** 544,773 trainable parameters (0.545 million) for 331 bands, 128 feature dimensions and five classes, using 1 × 1 adaptive average pooling. This release changes the previous package’s 4 × 4 pooling and spatial projection; checkpoints from that package are incompatible. The release is a manuscript-aligned implementation, not a verification of the architecture used to generate the reported experimental results.

**Important:** early checkpoints named `mcl_asf_sc_v2_best.pth` are *not architecturally compatible* with this paper-aligned model. See `docs/RELEASE_NOTES.md`.

## Citation and license

Manuscript title: *MCL-ASF: Ground-based Hyperspectral Small-Target Detection via Multi-Perturbation Spectral-Spatial Complementary Representation*.

Publication information, DOI and license will be filled in after author confirmation. No license is granted through this draft repository by default.
