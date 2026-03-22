"""
Differentiable JPEG compression simulation.

Implements the full JPEG pipeline (color convert → DCT → quantize → IDCT → color convert)
with a straight-through estimator (STE) for the quantization step, making the entire
pipeline differentiable for end-to-end training.

The STE trick: forward pass rounds normally, backward pass acts like identity.
    quantized = x + (torch.round(x) - x).detach()

Reference: Shin & Song, "JPEG-resistant Adversarial Images", NeurIPS 2017 Workshop
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


# ── Standard JPEG luminance quantization table ────────────────────────────────
# This is the baseline table from the JPEG standard (ITU-T T.81, Annex K).
# Quality scaling is applied at runtime by multiplying this table.
_JPEG_QUANT_TABLE = torch.tensor([
    [16, 11, 10, 16,  24,  40,  51,  61],
    [12, 12, 14, 19,  26,  58,  60,  55],
    [14, 13, 16, 24,  40,  57,  69,  56],
    [14, 17, 22, 29,  51,  87,  80,  62],
    [18, 22, 37, 56,  68, 109, 103,  77],
    [24, 35, 55, 64,  81, 104, 113,  92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103,  99],
], dtype=torch.float32)


def _quality_to_scale(quality: float) -> float:
    """
    Convert JPEG quality factor (1-100) to quantization scale.

    This matches libjpeg's quality-to-scale mapping:
        q < 50  → scale = 5000 / q
        q >= 50 → scale = 200 - 2*q

    Lower quality = higher scale = more quantization = more information loss.
    """
    if quality < 50:
        return 5000.0 / quality
    else:
        return 200.0 - 2.0 * quality


def _get_scaled_quant_table(quality: float, device: torch.device) -> torch.Tensor:
    """
    Scale the standard quantization table by the quality factor.

    Returns:
        (8, 8) quantization table, clamped to [1, 255].
    """
    scale = _quality_to_scale(quality)
    table = (_JPEG_QUANT_TABLE * scale / 100.0).clamp(1.0, 255.0)
    return table.to(device)


def _create_dct_matrix(device: torch.device) -> torch.Tensor:
    """
    Create the 8×8 DCT-II basis matrix.

    This is the orthonormal DCT matrix where entry (k, n) is:
        alpha(k) * cos(pi * (2n + 1) * k / 16)

    Using a precomputed matrix instead of torch.fft because:
    1. The DCT is always 8×8, so a small matrix multiply is faster than FFT overhead.
    2. It's trivially differentiable (just a linear transform).
    """
    n = 8
    dct_mat = torch.zeros(n, n, device=device)
    for k in range(n):
        for i in range(n):
            if k == 0:
                alpha = np.sqrt(1.0 / n)
            else:
                alpha = np.sqrt(2.0 / n)
            dct_mat[k, i] = alpha * np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    return dct_mat


class DifferentiableJPEG(nn.Module):
    """
    Differentiable JPEG compression for end-to-end training.

    Architecture:
        1. Split image into 8×8 blocks
        2. Apply DCT to each block (matrix multiply, fully differentiable)
        3. Quantize with STE (forward: round, backward: identity)
        4. Dequantize (multiply back by quant table)
        5. Apply IDCT (matrix multiply)
        6. Reassemble image from blocks

    The quality factor controls compression strength — lower quality means
    more aggressive quantization and more information loss.
    """

    def __init__(self) -> None:
        super().__init__()
        # DCT matrix is computed lazily on first forward pass to get correct device
        self._dct_matrix: Optional[torch.Tensor] = None

    def _get_dct_matrix(self, device: torch.device) -> torch.Tensor:
        """Lazy-init DCT matrix on the correct device."""
        if self._dct_matrix is None or self._dct_matrix.device != device:
            self._dct_matrix = _create_dct_matrix(device)
        return self._dct_matrix

    def _blockify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split image into non-overlapping 8×8 blocks.

        Input:  (B, C, H, W) where H, W are divisible by 8
        Output: (B*C*num_blocks, 8, 8)
        """
        B, C, H, W = x.shape
        # Reshape to extract 8×8 patches
        x = x.view(B * C, 1, H, W)
        # unfold extracts patches along each spatial dim
        patches = x.unfold(2, 8, 8).unfold(3, 8, 8)  # (B*C, 1, H//8, W//8, 8, 8)
        patches = patches.contiguous().view(-1, 8, 8)   # (num_total_blocks, 8, 8)
        return patches

    def _deblockify(self, patches: torch.Tensor, B: int, C: int,
                    H: int, W: int) -> torch.Tensor:
        """
        Reassemble 8×8 blocks back into a full image.

        Input:  (num_total_blocks, 8, 8)
        Output: (B, C, H, W)
        """
        bh, bw = H // 8, W // 8
        # Reshape: (B*C, bh, bw, 8, 8) → permute → (B*C, bh*8, bw*8) → (B, C, H, W)
        patches = patches.view(B * C, bh, bw, 8, 8)
        patches = patches.permute(0, 1, 3, 2, 4).contiguous()
        patches = patches.view(B, C, H, W)
        return patches

    def forward(self, image: torch.Tensor, quality: float = 75.0) -> torch.Tensor:
        """
        Apply differentiable JPEG compression.

        Args:
            image:   (B, C, H, W) in [0, 1]. H, W must be divisible by 8.
            quality: JPEG quality factor in [1, 100]. Lower = more compression.

        Returns:
            (B, C, H, W) JPEG-compressed image in [0, 1].
        """
        B, C, H, W = image.shape
        device = image.device

        # Pad if needed to make H, W divisible by 8
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        if pad_h > 0 or pad_w > 0:
            image = F.pad(image, (0, pad_w, 0, pad_h), mode='reflect')
            _, _, H_pad, W_pad = image.shape
        else:
            H_pad, W_pad = H, W

        # Shift from [0, 1] to [-0.5, 0.5] (JPEG operates around zero)
        x = image - 0.5

        # Get DCT matrix and quantization table
        dct = self._get_dct_matrix(device)           # (8, 8)
        qtable = _get_scaled_quant_table(quality, device)  # (8, 8)

        # Split into 8×8 blocks
        blocks = self._blockify(x)  # (N, 8, 8)

        # Forward DCT: D @ block @ D^T
        dct_blocks = torch.matmul(dct, torch.matmul(blocks, dct.t()))

        # Quantize with straight-through estimator
        # Forward: round(dct / qtable) * qtable (standard JPEG)
        # Backward: gradient flows through as if rounding didn't happen
        normalized = dct_blocks / qtable.unsqueeze(0)
        # STE: the .detach() on the rounding residual means no gradient through round()
        quantized = normalized + (torch.round(normalized) - normalized).detach()
        dequantized = quantized * qtable.unsqueeze(0)

        # Inverse DCT: D^T @ block @ D
        spatial_blocks = torch.matmul(dct.t(), torch.matmul(dequantized, dct))

        # Reassemble
        result = self._deblockify(spatial_blocks, B, C, H_pad, W_pad)

        # Shift back to [0, 1] and clamp
        result = (result + 0.5).clamp(0.0, 1.0)

        # Remove padding if added
        if pad_h > 0 or pad_w > 0:
            result = result[:, :, :H, :W]

        return result
