"""
DWT-based watermark decoder with channel-only attention (CBAM-inspired).

Pipeline:
    Input image → STN geometric correction → DWT → extract LL subband →
    residual CNN with channel attention → global avg pool → FC → 256 logits

NOTE: Only channel attention is used. Spatial attention is intentionally omitted
because it would conflict with the STN's geometric correction — both would try
to learn spatial transformations, creating competing gradients.

Install: pip install pytorch-wavelets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from .haar_dwt import HaarDWT
from typing import Optional

from .stn import SpatialTransformerNetwork


class ChannelAttention(nn.Module):
    """
    Channel attention module (CBAM channel branch only).

    Learns per-channel importance weights via squeeze-and-excitation:
        GAP + GMP → shared MLP → sigmoid → channel-wise scaling.

    Spatial attention is NOT included because the STN already handles
    spatial correction — adding spatial attention here would create
    conflicting spatial gradients during backprop.

    Reference: Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(channels // reduction, 4)  # Floor at 4 to prevent bottleneck collapse

        # Shared MLP applied to both GAP and GMP features
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) feature map.
        Returns:
            (B, C, H, W) attention-weighted feature map.
        """
        B, C, H, W = x.shape

        # Squeeze: global average pool + global max pool → (B, C)
        avg_feat = x.mean(dim=[2, 3])                     # (B, C)
        max_feat = x.flatten(2).max(dim=2)[0]              # (B, C)

        # Excitation: shared MLP on both, then add and sigmoid
        attn = torch.sigmoid(self.mlp(avg_feat) + self.mlp(max_feat))  # (B, C)

        # Scale: broadcast channel weights across spatial dims
        return x * attn.view(B, C, 1, 1)


class ResidualBlockWithAttention(nn.Module):
    """
    Pre-activation residual block with channel-only attention at the output.

    Architecture: BN→ReLU→Conv→BN→ReLU→Conv + skip → ChannelAttention
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )
        self.channel_attn = ChannelAttention(channels, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x + self.block(x)
        out = self.channel_attn(out)
        return out


class WatermarkDecoder(nn.Module):
    """
    Full decoder: extracts 256-bit watermark logits from a (possibly attacked) image.

    Steps:
        1. STN: learn affine correction for geometric distortions
        2. DWT: 1-level Haar on all channels, take LL subband
        3. Residual CNN with channel attention: extract watermark features
        4. Global average pool → FC → 256 logits (pre-sigmoid)
    """

    def __init__(
        self,
        watermark_length: int = 256,
        num_blocks: int = 3,
        filters: int = 64,
        attention_reduction: int = 16,
        use_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.use_checkpointing = use_checkpointing

        # ── Geometric correction ────────────────────────────────────────────────
        self.stn = SpatialTransformerNetwork(in_channels=3)

        # ── DWT ─────────────────────────────────────────────────────────────────
        # Pure-PyTorch Haar DWT — same config as encoder.
        self.dwt = HaarDWT()

        # ── Feature extraction CNN ──────────────────────────────────────────────
        # Input is LL subband: 3 channels (Y, Cb, Cr) at 128×128
        self.entry = nn.Sequential(
            nn.Conv2d(3, filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
        )

        self.res_blocks = nn.ModuleList([
            ResidualBlockWithAttention(filters, attention_reduction)
            for _ in range(num_blocks)
        ])

        # ── Classification head ─────────────────────────────────────────────────
        # GAP collapses spatial dims → FC maps to watermark logits
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),    # (B, filters, 1, 1)
            nn.Flatten(),               # (B, filters)
            nn.Linear(filters, watermark_length),  # (B, 256) — logits, no sigmoid
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract watermark logits from possibly-attacked watermarked image.

        Args:
            image: (B, 3, 256, 256) RGB watermarked image in [0, 1].

        Returns:
            (B, 256) watermark logits (pre-sigmoid). Apply sigmoid + threshold
            at 0.5 to recover binary bits at inference time.
        """
        # ── Step 1: Geometric correction via STN ────────────────────────────────
        corrected = self.stn(image)  # (B, 3, 256, 256)

        # ── Step 2: DWT → LL subband ────────────────────────────────────────────
        yl, _ = self.dwt(corrected)  # yl: (B, 3, 128, 128)

        # ── Step 3: Feature extraction with channel attention ───────────────────
        x = self.entry(yl)  # (B, filters, 128, 128)

        for block in self.res_blocks:
            if self.use_checkpointing and self.training:
                x = grad_checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        # ── Step 4: Global pool → FC → logits ──────────────────────────────────
        logits = self.head(x)  # (B, 256)
        return logits
