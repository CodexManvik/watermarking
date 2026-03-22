# Deep Learning Watermarking System (PyTorch)

Invisible 256-bit binary watermarking for RGB images using a DWT-based
encoder-decoder architecture trained end-to-end against a multi-branch
attack simulator (MBRS-style) with TrustMark loss improvements.

## Architecture

```
                     ┌──────────────┐
Image (3×256×256) ──►│   Encoder    │──► Watermarked Image (3×256×256)
Watermark (256,)  ──►│  (DWT+CNN)   │──► Delta (3×128×128)
                     └──────────────┘
                            │
                     ┌──────▼───────┐
                     │   Attack     │  ← 8 simultaneous branches
                     │  Simulator   │    (identity, JPEG, noise, etc.)
                     └──────┬───────┘
                            │
                     ┌──────▼───────┐
                     │   Decoder    │──► Recovered Watermark (256,)
                     │ (STN+DWT+CNN)│
                     └──────────────┘
```

### Encoder Pipeline
RGB → YCbCr → DWT (Haar) → extract LL subbands →
concatenate with upsampled watermark → residual CNN → delta →
texture-adaptive scaling → YCbCr 4:1:1 gain → IDWT → RGB

### Decoder Pipeline
Image → STN (affine correction) → DWT → LL subband →
residual CNN + channel attention → global avg pool → FC → 256 logits

### Loss Function
`λ1*MSE + λ2*LPIPS + λ3*(1-SSIM) + λ4*BCE(smoothed) + λ5*delta_L2`

### Curriculum Training
| Phase    | Epochs | Active Attacks                    |
|----------|--------|-----------------------------------|
| Phase 1  | 1–10   | Identity only (clean embedding)   |
| Phase 2  | 11–20  | + Gaussian noise, mild JPEG       |
| Phase 3  | 21–40  | Full suite, moderate strength     |
| Phase 4  | 41+    | Full suite, full strength         |

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training
```bash
# Full training
python trainer.py --data-dir ./train_images --epochs 100

# Smoke test (5 epochs)
python trainer.py --data-dir ./train_images --epochs 5

# Resume from checkpoint
python trainer.py --data-dir ./train_images --resume checkpoints/best_loss.pt
```

### Evaluation
```bash
python evaluate.py --checkpoint checkpoints/best_loss.pt --data-dir ./test_images
```

## Project Structure

```
watermarking/
├── config.py              # All hyperparameters and hardware flags
├── models/
│   ├── encoder.py         # DWT-based encoder with watermark embedding
│   ├── decoder.py         # DWT decoder with channel attention
│   └── stn.py             # Spatial Transformer Network
├── attacks/
│   ├── simulator.py       # 8-branch MBRS attack simulator
│   └── jpeg_diff.py       # Differentiable JPEG compression
├── losses.py              # Combined loss (MSE+LPIPS+SSIM+BCE+L2)
├── data/
│   └── loader.py          # Dataset and DataLoader
├── trainer.py             # Training loop with curriculum
├── evaluate.py            # Per-attack evaluation table
├── requirements.txt       # Dependencies
└── README.md              # This file
```

## Hardware

- **Primary**: RTX 3050 (~3.5 GB VRAM), batch size 16, no AMP
- **Kaggle**: P100/T4, batch size 32, AMP enabled
- Set `KAGGLE_MODE = True` in `config.py` for Kaggle

## References

- MBRS: Li et al., "MBRS: Enhancing Robustness of DNN-Based Watermarking by Mini-Batch of Real and Simulated JPEG Compression"
- TrustMark: Bui et al., "TrustMark: Universal Watermarking for Arbitrary Resolution Images"
- CBAM: Woo et al., "CBAM: Convolutional Block Attention Module"
- STN: Jaderberg et al., "Spatial Transformer Networks"
