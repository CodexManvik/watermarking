"""
Combined loss function for the watermarking system (TrustMark-inspired).

Total loss = λ1*MSE + λ2*LPIPS + λ3*(1-SSIM) + λ4*BCE(smoothed) + λ5*delta_L2

Components:
    - MSE:   pixel-level fidelity between original and watermarked image
    - LPIPS: perceptual quality via frozen VGG16 features
    - SSIM:  structural similarity (from pytorch-msssim)
    - BCE:   watermark bit recovery with label smoothing
    - L2:    delta regularization to keep the embedding perturbation small

Install:
    pip install lpips pytorch-msssim
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
from pytorch_msssim import ssim
from typing import Dict, Tuple


class WatermarkLoss(nn.Module):
    """
    Multi-component loss for joint image fidelity and watermark recovery.

    The loss balances five objectives:
        1. Pixel fidelity (MSE) — keeps the watermarked image close to original
        2. Perceptual quality (LPIPS) — ensures perceptual similarity via deep features
        3. Structural similarity (SSIM) — preserves local structure and contrast
        4. Watermark recovery (BCE) — trains decoder to recover embedded bits
        5. Delta regularization (L2) — penalizes large embedding perturbations

    Label smoothing on BCE targets (0→0.05, 1→0.95) prevents the decoder from
    producing overconfident logits that cause vanishing gradients on correctly-
    predicted bits. This is especially important in early training when the
    encoder hasn't yet learned stable embeddings.
    """

    def __init__(
        self,
        lambda_mse: float = 1.0,
        lambda_lpips: float = 0.5,
        lambda_ssim: float = 0.5,
        lambda_bce: float = 3.0,
        lambda_delta: float = 0.1,
        label_smooth_pos: float = 0.95,
        label_smooth_neg: float = 0.05,
        device: str = "cpu",
        # Bootstrap phase: zero out image quality losses for the first N epochs
        # so the encoder/decoder can co-adapt without MSE/SSIM suppressing embedding.
        # This is the standard trick used in HiDDeN, SteganoGAN, and related work.
        bootstrap_epochs: int = 5,
        # Minimum embedding strength: add a penalty if the delta RMS falls below
        # this value, preventing the trivial solution where the encoder embeds nothing.
        min_embed_rms: float = 0.05,
    ) -> None:
        super().__init__()
        self.lambda_mse = lambda_mse
        self.lambda_lpips = lambda_lpips
        self.lambda_ssim = lambda_ssim
        self.lambda_bce = lambda_bce
        self.lambda_delta = lambda_delta
        self.label_smooth_pos = label_smooth_pos
        self.label_smooth_neg = label_smooth_neg
        self.bootstrap_epochs = bootstrap_epochs
        self.min_embed_rms = min_embed_rms

        # ── LPIPS (frozen VGG16) ────────────────────────────────────────────────
        # The LPIPS network is fully frozen: it only extracts features for the
        # perceptual loss, never receives gradient updates.  Using VGG (not Alex)
        # because VGG features correlate better with human perceptual judgments.
        self.lpips_fn = lpips.LPIPS(net='vgg', verbose=False).to(device)
        self.lpips_fn.eval()
        for param in self.lpips_fn.parameters():
            param.requires_grad = False

    def _smooth_targets(self, watermark: torch.Tensor) -> torch.Tensor:
        """
        Apply label smoothing to binary watermark targets.

        Hard targets 0/1 are mapped to soft targets 0.05/0.95.
        This prevents the BCE loss from driving logits to ±infinity,
        which would cause vanishing gradients for correctly-predicted bits.
        """
        return watermark * self.label_smooth_pos + (1 - watermark) * self.label_smooth_neg

    def compute_quality(
        self,
        original_image: torch.Tensor,
        watermarked_image: torch.Tensor,
        delta: torch.Tensor,
        epoch: int = 1,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total quality loss and individual component values.
        """
        bootstrapping = epoch <= self.bootstrap_epochs
        ramp_epochs = 5
        if bootstrapping:
            quality_weight = 0.0
        elif epoch <= self.bootstrap_epochs + ramp_epochs:
            quality_weight = (epoch - self.bootstrap_epochs) / ramp_epochs
        else:
            quality_weight = 1.0

        # ── MSE loss ───────────────────────────────────────────────────────────
        mse_loss = F.mse_loss(watermarked_image, original_image) if quality_weight > 0 \
            else torch.tensor(0.0, device=original_image.device)

        # ── LPIPS loss ──────────────────────────────────────────────────────────
        def _lpips_safe(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            """Compute LPIPS in small sub-batches at half resolution."""
            a_d = F.interpolate(a, size=(128, 128),
                                mode='bilinear', align_corners=False)
            b_d = F.interpolate(b.detach(), size=(128, 128),
                                mode='bilinear', align_corners=False)
            a_d = a_d * 2 - 1
            b_d = b_d * 2 - 1
            chunk = 4
            vals = []
            for i in range(0, a_d.size(0), chunk):
                vals.append(self.lpips_fn(a_d[i:i+chunk], b_d[i:i+chunk]).mean())
            return torch.stack(vals).mean()

        if quality_weight > 0:
            lpips_loss = _lpips_safe(watermarked_image, original_image)
        else:
            lpips_loss = torch.tensor(0.0, device=original_image.device)

        # ── SSIM loss ──────────────────────────────────────────────────────────
        if quality_weight > 0:
            ssim_val = ssim(watermarked_image, original_image,
                            data_range=1.0, size_average=True)
            ssim_loss = 1.0 - ssim_val
        else:
            ssim_val = torch.tensor(0.0, device=original_image.device)
            ssim_loss = torch.tensor(0.0, device=original_image.device)

        # ── Delta L2 regularization ──────────────────────────────────────────────────
        delta_loss = torch.mean(delta ** 2)

        # ── Minimum embedding strength ──────────────────────────────────────────────────
        embed_rms = torch.sqrt(torch.mean(delta ** 2) + 1e-8)
        min_embed_loss = F.relu(self.min_embed_rms - embed_rms)

        # ── Total loss ──────────────────────────────────────────────────────────────
        total = (
            quality_weight * self.lambda_mse   * mse_loss
            + quality_weight * self.lambda_lpips * lpips_loss
            + quality_weight * self.lambda_ssim  * ssim_loss
            + self.lambda_delta                  * delta_loss
            + 10.0                               * min_embed_loss
        )

        components = {
            "mse":       mse_loss.item(),
            "lpips":     lpips_loss.item(),
            "ssim":      ssim_val.item() if quality_weight > 0 else 0.0,
            "delta_l2":  delta_loss.item(),
            "embed_rms": embed_rms.item(),
        }

        return total, components

    def compute_bce(
        self,
        watermark_logits: torch.Tensor,
        watermark_target: torch.Tensor,
        epoch: int = 1,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        bootstrapping = epoch <= self.bootstrap_epochs
        ramp_epochs = 5
        if bootstrapping:
            bce_weight = self.lambda_bce * 2.0
        elif epoch <= self.bootstrap_epochs + ramp_epochs:
            bce_weight = self.lambda_bce
        else:
            bce_weight = self.lambda_bce

        smooth_targets = self._smooth_targets(watermark_target)
        bce_loss = F.binary_cross_entropy_with_logits(
            watermark_logits, smooth_targets
        )

        total = bce_weight * bce_loss
        components = {
            "bce": bce_loss.item(),
        }

        return total, components
