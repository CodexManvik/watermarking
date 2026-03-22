"""
Lightweight Spatial Transformer Network (STN) for geometric correction.

The STN learns to undo geometric distortions (rotation, translation, scale)
applied by the attack simulator, making the decoder robust to geometric attacks.
Only an affine transform is used — no deformable grids.

Reference: Jaderberg et al., "Spatial Transformer Networks", NeurIPS 2015.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SpatialTransformerNetwork(nn.Module):
    """
    Affine STN that learns to correct geometric distortions in the input image.

    Architecture:
        Localization CNN → FC layers → 6-parameter affine matrix → grid_sample

    The localization net is intentionally small (3 conv layers) because:
    1. It only needs to estimate 6 affine parameters, not pixel-level features.
    2. A large localizer would consume VRAM disproportionate to its contribution.
    3. The STN is a preprocessing step — the decoder CNN does the heavy lifting.
    """

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()

        # ── Localization network ────────────────────────────────────────────────
        # Small CNN that regresses the 6 affine parameters from the input image.
        # Pooling is aggressive (stride-2 convs + max pools) to reduce spatial
        # dimensions quickly — we only need a global descriptor, not spatial detail.
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),  # 256 → 64

            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),  # 64 → 16

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),    # → (B, 64, 1, 1)
        )

        # ── Regression head ─────────────────────────────────────────────────────
        # Maps the 64-dim feature vector to 6 affine parameters.
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 6),
        )

        # ── Identity initialization ─────────────────────────────────────────────
        # Critical: initialize the affine output to identity so the STN starts
        # as a no-op. Without this, random initial transforms would destroy the
        # image and prevent the decoder from learning anything in early epochs.
        self.fc[-1].weight.data.zero_()
        self.fc[-1].bias.data.copy_(
            torch.tensor([1.0, 0.0, 0.0,
                          0.0, 1.0, 0.0], dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply learned affine correction to input image.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Geometrically corrected tensor, same shape as input.
        """
        B = x.size(0)

        # Localization: extract global feature → regress affine params
        features = self.localization(x).view(B, -1)  # (B, 64)
        theta = self.fc(features).view(B, 2, 3)       # (B, 2, 3) affine matrix

        # Generate sampling grid and apply affine transform
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        # align_corners=False is the PyTorch default and matches the behavior
        # expected by grid coordinates in [-1, 1] range.
        corrected = F.grid_sample(x, grid, align_corners=False,
                                  mode='bilinear', padding_mode='border')
        # padding_mode='border' avoids black borders on corrected images,
        # which would create artifacts in the decoder's DWT output.

        return corrected
