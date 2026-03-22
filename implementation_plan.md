# PyTorch Watermarking System — Full Rebuild

Rebuild the existing TensorFlow DWT-based watermarking system in pure PyTorch. The system embeds invisible 256-bit binary watermarks into RGB images using a DWT-based encoder-decoder architecture trained end-to-end against a multi-branch attack simulator (MBRS-style), with TrustMark loss improvements.

## Proposed Changes

### Configuration

#### [NEW] [config.py](file:///home/manvik/Desktop/new_crypt/watermarking/config.py)
All hyperparameters in one place: image size (256×256), watermark length (256), batch sizes (16 / 32 for Kaggle), learning rate (1e-4), loss weights (λ1–λ5), curriculum epoch boundaries, `KAGGLE_MODE` flag, paths for data/checkpoints, and device selection.

---

### Models

#### [NEW] [stn.py](file:///home/manvik/Desktop/new_crypt/watermarking/models/stn.py)
Lightweight Spatial Transformer Network for geometric correction. Uses a small localization CNN → FC layer producing a 6-parameter affine matrix, initialized to identity. Uses `F.grid_sample` for differentiable image warping.

#### [NEW] [encoder.py](file:///home/manvik/Desktop/new_crypt/watermarking/models/encoder.py)
- RGB → YCbCr conversion
- 1-level Haar DWT via `pytorch_wavelets` per channel
- Extract LL subbands (Y primary, Cb/Cr at 4:1:1 gain)
- Watermark upsampling: `(B,256)` → reshape `(B,1,16,16)` → 3 transposed conv layers → `(B,1,128,128)`
- Concatenate watermark with LL band → 3 residual blocks (64 filters) → delta perturbation with tanh
- Adaptive texture-based scaling (gradient magnitude map)
- Add delta to LL → IDWT → YCbCr → RGB
- Gradient checkpointing on residual blocks

#### [NEW] [decoder.py](file:///home/manvik/Desktop/new_crypt/watermarking/models/decoder.py)
- Input through STN for geometric correction
- DWT → extract LL subband
- 3 residual blocks (64 filters) with CBAM-style channel attention
- Global average pool → FC → 256-dim logits (pre-sigmoid)

#### [NEW] [\_\_init\_\_.py](file:///home/manvik/Desktop/new_crypt/watermarking/models/__init__.py)
Exports `WatermarkEncoder`, `WatermarkDecoder`.

---

### Attacks

#### [NEW] [jpeg_diff.py](file:///home/manvik/Desktop/new_crypt/watermarking/attacks/jpeg_diff.py)
Differentiable JPEG simulation — DCT/IDCT with quantization using the straight-through estimator (`x + (round(x) - x).detach()`). Quality factor as input parameter.

#### [NEW] [simulator.py](file:///home/manvik/Desktop/new_crypt/watermarking/attacks/simulator.py)
8-branch attack simulator applied **simultaneously** (all branches, loss averaged):
- Identity, Diff-JPEG, Gaussian noise, Salt-and-pepper, Pixel dropout, Rotation (`grid_sample`), Average blur, Screenshotting composite
- Takes `current_epoch` and applies curriculum-based strength scaling

#### [NEW] [\_\_init\_\_.py](file:///home/manvik/Desktop/new_crypt/watermarking/attacks/__init__.py)
Exports `AttackSimulator`.

---

### Losses

#### [NEW] [losses.py](file:///home/manvik/Desktop/new_crypt/watermarking/losses.py)
Combined loss: `λ1*MSE + λ2*LPIPS + λ3*SSIM_loss + λ4*BCE(smoothed) + λ5*delta_L2`.
- LPIPS using frozen VGG16 from `lpips` library
- SSIM from `pytorch-msssim`
- BCE with label smoothing (0→0.05, 1→0.95)

---

### Data

#### [NEW] [loader.py](file:///home/manvik/Desktop/new_crypt/watermarking/data/loader.py)
- `WatermarkDataset`: loads images from a directory, resizes to 256×256, normalizes to [0,1], generates random 256-bit watermarks on the fly
- `get_dataloader()`: returns `DataLoader` with configurable batch size and workers

#### [NEW] [\_\_init\_\_.py](file:///home/manvik/Desktop/new_crypt/watermarking/data/__init__.py)
Exports `WatermarkDataset`, `get_dataloader`.

---

### Training & Evaluation

#### [NEW] [trainer.py](file:///home/manvik/Desktop/new_crypt/watermarking/trainer.py)
- Instantiates encoder, decoder, attack simulator, loss
- Adam optimizer (lr=1e-4, gradient clipping norm=1.0)
- Curriculum schedule: epochs 1–10 identity-only → 11–20 mild attacks → 21–40 moderate → 41+ full
- Per-epoch logging: total loss, PSNR, BER
- Checkpoint saving (best loss, best PSNR)
- `KAGGLE_MODE` branch for larger batch + mixed precision

#### [NEW] [evaluate.py](file:///home/manvik/Desktop/new_crypt/watermarking/evaluate.py)
- Loads trained encoder + decoder weights
- Runs each attack individually at fixed strengths
- Prints a table: Attack | PSNR | SSIM | BER | Bit Accuracy

#### [NEW] [requirements.txt](file:///home/manvik/Desktop/new_crypt/watermarking/requirements.txt)
All pip dependencies.

#### [NEW] [README.md](file:///home/manvik/Desktop/new_crypt/watermarking/README.md)
Project overview, architecture, install instructions, usage.

---

## Verification Plan

### Automated Tests (run sequentially after all files are written)

1. **Import check** — verify all modules import cleanly:
   ```bash
   cd /home/manvik/Desktop/new_crypt/watermarking
   python -c "from models.encoder import WatermarkEncoder; from models.decoder import WatermarkDecoder; from attacks.simulator import AttackSimulator; from losses import WatermarkLoss; from data.loader import get_dataloader; print('All imports OK')"
   ```

2. **Shape validation** — run a forward pass with dummy data and assert tensor shapes:
   ```bash
   cd /home/manvik/Desktop/new_crypt/watermarking
   python -c "
   import torch
   from models.encoder import WatermarkEncoder
   from models.decoder import WatermarkDecoder
   enc = WatermarkEncoder().to('cpu')
   dec = WatermarkDecoder().to('cpu')
   img = torch.rand(2,3,256,256)
   wm = torch.randint(0,2,(2,256)).float()
   wm_img, delta = enc(img, wm)
   assert wm_img.shape == (2,3,256,256), f'Bad enc shape: {wm_img.shape}'
   logits = dec(wm_img)
   assert logits.shape == (2,256), f'Bad dec shape: {logits.shape}'
   print('Shape validation PASSED')
   "
   ```

3. **Training smoke test** — run `trainer.py` for 5 epochs:
   ```bash
   cd /home/manvik/Desktop/new_crypt/watermarking
   python trainer.py --epochs 5 --data-dir <path_to_images>
   ```
   - Verify PSNR > 30 dB from epoch 1
   - Verify BER < 50% from epoch 1

4. **Evaluation test** — run `evaluate.py`:
   ```bash
   cd /home/manvik/Desktop/new_crypt/watermarking
   python evaluate.py --checkpoint <path>
   ```
   - Verify per-attack results table prints correctly

### Manual Verification
- User should visually inspect a watermarked image vs original to confirm invisibility
- User should check GPU memory usage stays within 3.5 GB on RTX 3050
