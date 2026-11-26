# Verification of release 0.1.1

- CPU tests executed with PyTorch 2.10.0.
- The default model (331 bands, 5 classes, 128 feature dimensions, 1 × 1 spatial adaptive pooling) has **544,773 trainable parameters** (0.545 million rounded).
- Synthetic forward/backward and ASF weight normalization: passed.
- Unit tests: 3/3 passed, including a 50 × 50 model input and exact parameter-count assertion.
- Synthetic training: 10 random patches, one epoch, checkpoint saved and reloaded successfully.
- Synthetic patch inference produced five class probabilities; synthetic 12 × 16 × 331 cube inference produced a finite [4, 12, 16] score map.
- No private dataset or pretrained weights are included.

Not verified: equivalence to the original experimental checkpoint, manuscript detection metrics, sensor-specific annotation processing, GPU timing, or identity of the generic cube inference pipeline with the original experiments. The change from 4 × 4 to 1 × 1 pooling invalidates earlier spatial projection weights.
