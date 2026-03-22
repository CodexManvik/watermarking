"""
Global configuration for the PyTorch watermarking system.
All hyperparameters, paths, and hardware flags live here.
"""

import torch
from dataclasses import dataclass, field
from typing import Tuple


# ── Hardware mode flag ──────────────────────────────────────────────────────────
# Set to True when running on Kaggle (P100/T4): enables larger batch and AMP.
KAGGLE_MODE: bool = False


@dataclass
class Config:
    """Central config object — instantiate once, pass everywhere."""

    # ── Image / watermark dimensions ────────────────────────────────────────────
    image_size: Tuple[int, int] = (256, 256)
    watermark_length: int = 256

    # ── Training ────────────────────────────────────────────────────────────────
    # Batch size is hardware-dependent: 16 for RTX 3050 (~3.5 GB), 32 for Kaggle
    batch_size: int = 32 if KAGGLE_MODE else 16
    num_workers: int = 4
    epochs: int = 100
    learning_rate: float = 1e-4
    grad_clip_norm: float = 1.0

    # Mixed precision is only safe on Kaggle — pytorch_wavelets has issues with
    # float16 intermediates on the DWT path, so we disable AMP on local GPU.
    use_amp: bool = KAGGLE_MODE

    # ── Loss weights (TrustMark-inspired) ───────────────────────────────────────
    # λ4 is highest because watermark recovery is the primary objective.
    lambda_mse: float = 1.0       # λ1: pixel-level fidelity
    lambda_lpips: float = 0.5     # λ2: perceptual quality (frozen VGG16)
    lambda_ssim: float = 0.5      # λ3: structural similarity
    lambda_bce: float = 3.0       # λ4: watermark recovery — most important
    lambda_delta: float = 0.1     # λ5: delta regularization (keeps perturbation small)

    # BCE label smoothing: hard targets 0/1 → soft targets 0.05/0.95 to prevent
    # overconfident logits that cause vanishing gradients on correctly-predicted bits.
    label_smooth_pos: float = 0.95
    label_smooth_neg: float = 0.05

    # ── Encoder architecture ────────────────────────────────────────────────────
    encoder_residual_blocks: int = 3
    encoder_filters: int = 64
    delta_scale: float = 0.55     # Tanh output * delta_scale bounds max perturbation

    # YCbCr 4:1:1 gain ratio: Y channel gets full delta, Cb/Cr get 1/4.
    # This is applied as explicit multipliers on the embedding delta before IDWT,
    # NOT just as a comment — the multipliers directly scale the perturbation.
    ycbcr_gain: Tuple[float, float, float] = (1.0, 0.25, 0.25)

    # ── Decoder architecture ────────────────────────────────────────────────────
    decoder_residual_blocks: int = 3
    decoder_filters: int = 64
    # Channel attention reduction ratio for CBAM-style attention.
    # NOTE: Only channel attention is used — spatial attention is intentionally
    # omitted because it would conflict with the STN's geometric correction.
    attention_reduction: int = 16

    # ── Curriculum training schedule ────────────────────────────────────────────
    # Phase boundaries define when attack branches activate:
    #   Phase 1 (1–10):  identity only — learn clean embedding first
    #   Phase 2 (11–20): add Gaussian noise + mild JPEG (q=85)
    #   Phase 3 (21–40): full attack suite at moderate strength
    #   Phase 4 (41+):   full attack suite at full strength
    curriculum_phase1_end: int = 10
    curriculum_phase2_end: int = 20
    curriculum_phase3_end: int = 40

    # ── Paths ───────────────────────────────────────────────────────────────────
    train_data_dir: str = "/home/manvik/Desktop/watermarking/train_images/"
    val_data_dir: str = "/home/manvik/Desktop/watermarking/test_images/"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # ── Device ──────────────────────────────────────────────────────────────────
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    # ── Gradient checkpointing ──────────────────────────────────────────────────
    # Enable on residual blocks to trade compute for VRAM on RTX 3050.
    use_gradient_checkpointing: bool = not KAGGLE_MODE


def get_config() -> Config:
    """Factory: returns a Config with KAGGLE_MODE already baked in."""
    return Config()
