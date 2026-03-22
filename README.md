# Deep Learning Watermarking System: In-Depth Architecture & Mechanics

This repository contains a full PyTorch implementation of a 256-bit blind image watermarking system built around a Discrete Wavelet Transform (DWT), Spatial Transformer Networks (STN), and an MBRS-flavored continuous attack simulator pipeline.

This document serves as an exhaustive breakdown of the internal mechanics, calculations, and mathematical constraints employed within the model.

---

## 1. Encoder Engine Mechanics (`models/encoder.py`)

The Encoder accepts a batch of RGB images `[B, 3, 256, 256]` mapped to `[0, 1]` and a flat binary sequence `[B, 256]`. The goal is to imperceptibly modify the spatial geometry of the image so that the bits can survive destructive augmentations.

### 1.1 Colorspace Translation
Rather than embedding into RGB (which strongly correlates channels), the system strictly works within the un-correlated **YCbCr** domain using exact ITU-R BT.601 conversion scalars. 
- `Y (Luminance)`: Extracted via `0.299·R + 0.587·G + 0.114·B`
- `Cb/Cr (Chrominance)`: Shifted by `0.5` after scaling to center at `[0,1]`.

### 1.2 DWT Frequency Sub-Banding
Applying a **single-level Haar Discrete Wavelet Transform (DWT)** on the full image tensor produces four 128x128 frequency bands (per channel): `LL, LH, HL, HH`. The system isolates the `LL` (Low-Low) subbands for embedding. High frequencies (`LH`, `HL`, `HH`) are incredibly volatile—they get obliterated by basic JPEG compression or blur filters. In contrast, the `LL` band controls baseline structural energy and survives smoothing operators reliably.

### 1.3 Target Map Upsampling
The flat `[B, 256]` watermark requires a local spatial projection. It is expanded to `[B, 1, 16, 16]` without a dense layer (preserving intrinsic structure), then scaled up via **Transposed Convolutions**:
- `16x16 → 32x32 → 64x64 → 128x128`. This forces the network to learn how to spatially "smear" 256-bit vectors over small overlapping local grid neighborhoods.

### 1.4 Central CNN Residual Mapping
The `Y` channel `LL` subband concatenates with the new `128x128` watermark map, generating a `[B, 2, 128, 128]` state. This is fed to 3 Pre-Activation (`BN → ReLU → Conv2D`) Residual Layers. The final output generates the unscaled base perturbation variable `raw_delta [B, 1, 128, 128]`. A `Tanh()` activation permanently binds pixel modifications mathematically between `[-1.0, 1.0]`.

### 1.5 Texture-Adaptive Power Scaling
Injecting delta perturbations into flat areas (like a clear sky) creates noticeable artifacts. Edges and high-frequency textures naturally hide perturbations better. The network accounts for this by computing a runtime spatial density map:
1. Interpolates the image to 128x128 and grayscales it.
2. Derives spatial gradients matching a finite-difference Sobel-like approach (`gx` and `gy`).
3. Determines spatial magnitude: `magnitude = sqrt(gx² + gy² + 1e-8)`.
4. Normalizes this magnitude map batch-wise to `[0, 1]`.
5. Maps using a piecewise floor: `texture_map = 0.3 + 0.7 * normalized`. This guarantees that even perfectly smooth gradients still accept 30% of the normal perturbation weight.
Final perturbation: `scaled_delta = raw_delta * delta_scale(0.55) * texture_map`.

### 1.6 Chrominance Damping Factor and IDWT Rebuild
When pushing the delta back to the image bands, the model scales it according to a strict **4:1:1 gain curve** (`config.ycbcr_gain`):
- `Y` channel accepts `1.0x` scaled delta.
- `Cb` and `Cr` channels accept isolated `0.25x` scaled deltas.

The final values `yl_new = yl_old + scaled_gains` are merged with the original `yh` details via an inverse Haar Wavelet pass (**IDWT**) back to YCbCr, then un-tilted to standard RGB.

---

## 2. Multi-Branch Attack Simulator (`attacks/simulator.py`)

Robustness isn't a post-process; it is computed concurrently. Instead of optimizing against simulated attacks one at a time, the continuous Multi-Branch system slices through 8 variations per batch.

1. **Identity**: A control zero-state pipeline.
2. **Differentiable JPEG**: Custom compression simulating blocking artifacts in frequency space at uniformly sampled quality floors `U[40, 95]`.
3. **Gaussian Noise**: Mean shifted offsets using `sigma = U[0.01, 0.05 * strength]`. 
4. **Salt & Pepper**: Random bit-flipping masks targeting extreme thresholds. Uses density `U[0.01, 0.05 * strength]`, forcing 0.0 or 1.0 absolute pixels explicitly.
5. **Pixel Dropout**: Zeros out pixel indices permanently at rates `U[0.05, 0.15 * strength]`.
6. **Bilinear Rotations**: Geometric tests warping grids locally through PyTorch `F.affine_grid` and `grid_sample` logic rotated up to `±30 deg * strength`.
7. **Box Blur (Average)**: Uniform `3x3, 5x5, 7x7` independent depthwise channel averaging passes.
8. **Screenshot Simulation**: A continuous cascaded function combining JPEG compression `[q 70-85]` + deep Gaussian blurring (kernel=5, variable sigma) + Brightness/Contrast matrix jittering `[± 5%]`, successfully mimicking a physical smartphone camera picture of a monitor.

### 2.1 Curriculum Learning Engine
To prevent early-epoch collapses due to impossible decodings:
- **Phase 1 (Epochs 01-10)**: Only Identity is active. Let the model organically shape weights.
- **Phase 2 (Epochs 11-20)**: Linear noise additions + highly forgiving JPEG compression algorithms.
- **Phase 3 (Epochs 21-40)**: Complete array of all 8 attacks enabled at exactly `0.5x → 1.0x` scaling intensity linearly correlated directly to the epoch count relative distance.
- **Phase 4 (Epoch 40+)**: Full intensity parameters without restriction.

---

## 3. Decoder Recovery Operations (`models/decoder.py`)

1. **Spatial Transformer Network (STN)**: The absolute first operation mapping incoming degraded tensors evaluates whether a geometric correction is required (combatting crop + rotation attacks). The STN predicts a `2x3` dynamic affine matrix and reverse-warps the pixels into their original orientation cleanly grid-sampled to 256x256 before continuing.
2. **Frequency DWT Execution**: Repetition of Haar DWT to immediately strip unreliable high-frequency data from the recovery space.
3. **Channel Attention Blocks (CBAM Channel-only Variant)**: Squeezes features by mapping Global Average Pools (`GAP`) and Global Max Pools (`GMP`) linearly. Both arrays are combined across a bottlenecked linear dimension network `[channels, mid=max(channels//16, 4)]`, and squashed by Sigmoid logic. Spatial Attention layers are strictly avoided because they battle spatial gradient allocations with the STN's calculations, crashing descent logic.
4. **Logit Extractions**: Spatial coordinates are collapsed completely via Adaptive Average Pooling into raw nodes. A fully-connected tier maps probabilities out to a `[B, 256]` final pre-sigmoid state logit output.

---

## 4. Total Loss Equations (`losses.py`)

A meticulously blended metric based roughly on TrustMark paradigms, carefully scaling 5 specific vectors (and an implicit 6th):

### 4.1. Loss Vector Specifications
- **L1/MSE (Weight: λ1 = 1.0)**: `Mean Squared Error` enforcing strict absolute differences between RGB original frames and watermarked reconstructions. 
- **LPIPS (Weight: λ2 = 0.5)**: Deep Structural validation checking perceptual feature maps output by an internally frozen `VGG16` block. (Run dynamically scaled at 128x128 interpolations locally to protect smaller VRAM reservoirs like an RTX 3050 footprint).
- **SSIM (Weight: λ3 = 0.5)**: Checks Structural local variance values. Calculation computes `1.0 - measured_ssim()`.
- **BCE / Decoding Error (Weight: λ4 = 3.0)**: Uses Binary Cross Entropy applied softly to avoid overconfident node collapsing. `Label_Smoothing` forces 1 and 0 hard truths dynamically into targets of `0.95` and `0.05`.
- **Delta Regularization (Weight: λ5 = 0.1)**: L2 penalty checking `mean(delta ** 2)`, actively restricting the network from exploiting infinite spatial boundaries.

### 4.2. Bootstrap Training Control
For the absolute first 5 epochs (`bootstrap_epochs`), `MSE, LPIPS, and SSIM are overridden entirely functionally to 0.0`. Simultaneously, BCE (Error) is spiked from a relative weight of `3.0` straight effectively to `6.0`. When `Epoch > 5`, image fidelity losses are linearly bridged from `0% to 100% influence` over the following 5 epochs. Why? Because image fidelity and watermark embedding are adversarial vectors. If both start immediately, the model defaults into a local minimum where it completely ignores watermarking, settling for perfect visual quality and failing payload delivery entirely. Bootstrapping locks in a stable structural modification pathway before demanding optimization.

### 4.3. Anti-Bypass Mechanisms
A calculated Minimum Embed Penalty exists to permanently kill situations during quality-loss tuning where the delta falls below visually noticeable amounts by outputting noise thresholds below recovery limit:
```python
embed_rms = sqrt(mean(delta ** 2) + 1e-8)
min_embed_loss = F.relu(0.005 - embed_rms) * 10.0
```
This forces the network's `delta RMS` to at least sit above the `0.005` scale. If smaller, a severe linear penalty multiplier of `10.0` engages proportionally ensuring stability in generating actual watermarks.
