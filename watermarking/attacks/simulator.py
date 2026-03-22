"""
Multi-Branch Attack Simulator (MBRS-style).

All 8 attack branches run simultaneously during training (not one-at-a-time),
and the training loss is averaged across all branches. This forces the encoder
to produce watermarks robust to ALL attacks at once.

The current_epoch parameter drives a curriculum schedule that gradually
introduces harder attacks as training progresses.

Branches:
    0: Identity (no attack)
    1: Differentiable JPEG — quality U(40, 95) per batch
    2: Gaussian noise — sigma U(0.01, 0.05)
    3: Salt and pepper noise — density U(0.01, 0.05)
    4: Pixel dropout — rate U(0.05, 0.15)
    5: Random rotation — angle U(-30°, 30°) via grid_sample
    6: Average blur — kernel U(3, 7) odd only
    7: Screenshotting simulation — JPEG + Gaussian blur + brightness/contrast jitter
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from typing import List, Tuple, Optional

from .jpeg_diff import DifferentiableJPEG


class AttackSimulator(nn.Module):
    """
    MBRS-style multi-branch attack simulator with curriculum scheduling.

    All branches execute simultaneously during training. The loss from each
    branch is averaged, forcing the encoder/decoder to be robust to all attacks
    jointly rather than specializing for one at a time.
    """

    def __init__(
        self,
        curriculum_phase1_end: int = 10,
        curriculum_phase2_end: int = 20,
        curriculum_phase3_end: int = 40,
    ) -> None:
        super().__init__()
        self.phase1_end = curriculum_phase1_end
        self.phase2_end = curriculum_phase2_end
        self.phase3_end = curriculum_phase3_end

        self.diff_jpeg = DifferentiableJPEG()

        # Branch names for logging / evaluation
        self.branch_names: List[str] = [
            "identity",
            "jpeg",
            "gaussian_noise",
            "salt_pepper",
            "pixel_dropout",
            "rotation",
            "avg_blur",
            "screenshot",
        ]

    def _get_active_branches(self, epoch: int) -> List[int]:
        """
        Curriculum schedule: which branches are active at this epoch.

        Phase 1 (1–phase1_end):  Identity only — learn clean embedding first
        Phase 2 (phase1_end+1–phase2_end): + Gaussian noise, mild JPEG
        Phase 3 (phase2_end+1–phase3_end): All branches at moderate strength
        Phase 4 (phase3_end+1+): All branches at full strength
        """
        if epoch <= self.phase1_end:
            return [0]  # Identity only
        elif epoch <= self.phase2_end:
            return [0, 1, 2]  # Identity + JPEG + Gaussian noise
        else:
            return list(range(8))  # All branches

    def _get_strength_multiplier(self, epoch: int) -> float:
        """
        Scale attack strength based on training phase.

        Phase 3: moderate strength (0.5× to 1.0× linear ramp)
        Phase 4: full strength (1.0×)
        Phases 1-2: only mild attacks active, so multiplier doesn't matter much
        """
        if epoch <= self.phase2_end:
            return 0.5  # Mild strength for early phases
        elif epoch <= self.phase3_end:
            # Linear ramp from 0.5 to 1.0 over phase 3
            progress = (epoch - self.phase2_end) / (self.phase3_end - self.phase2_end)
            return 0.5 + 0.5 * progress
        else:
            return 1.0  # Full strength

    # ── Individual attack branches ──────────────────────────────────────────────

    def _identity(self, image: torch.Tensor) -> torch.Tensor:
        """Branch 0: pass-through, no modification."""
        return image

    def _jpeg(self, image: torch.Tensor, strength: float) -> torch.Tensor:
        """
        Branch 1: Differentiable JPEG compression.

        Quality sampled from U(40, 95). At lower strength, quality floor is raised
        to reduce the attack's impact during early curriculum phases.
        """
        # Interpolate quality range based on strength:
        # strength=0.5 → q ∈ [70, 95], strength=1.0 → q ∈ [40, 95]
        q_low = 95 - strength * 55  # 0.5→67.5, 1.0→40
        q_high = 95.0
        quality = random.uniform(q_low, q_high)
        return self.diff_jpeg(image, quality=quality)

    def _gaussian_noise(self, image: torch.Tensor, strength: float) -> torch.Tensor:
        """
        Branch 2: Additive Gaussian noise.

        Sigma sampled from U(0.01, 0.05). Strength scales the upper bound.
        """
        sigma_max = 0.05 * strength
        sigma = random.uniform(0.01, max(0.01, sigma_max))
        noise = torch.randn_like(image) * sigma
        return (image + noise).clamp(0.0, 1.0)

    def _salt_pepper(self, image: torch.Tensor, strength: float) -> torch.Tensor:
        """
        Branch 3: Salt-and-pepper noise.

        Density sampled from U(0.01, 0.05). Each pixel independently becomes
        salt (1.0) or pepper (0.0) with equal probability.
        """
        density = random.uniform(0.01, 0.05 * strength)
        # Generate noise mask
        rand_map = torch.rand_like(image)
        salt_mask = (rand_map < density / 2).float()
        pepper_mask = (rand_map > 1.0 - density / 2).float()
        # Apply: salt pixels → 1.0, pepper pixels → 0.0, rest unchanged
        result = image * (1 - salt_mask) * (1 - pepper_mask) + salt_mask
        return result

    def _pixel_dropout(self, image: torch.Tensor, strength: float) -> torch.Tensor:
        """
        Branch 4: Random pixel dropout (zeroing).

        Rate sampled from U(0.05, 0.15). Each pixel independently zeroed.
        """
        rate = random.uniform(0.05, 0.15 * strength)
        mask = (torch.rand_like(image) > rate).float()
        return image * mask

    def _rotation(self, image: torch.Tensor, strength: float) -> torch.Tensor:
        """
        Branch 5: Random rotation via grid_sample.

        Angle sampled from U(-30°, 30°). Uses bilinear interpolation and
        border padding (not zero padding, which would create black corners
        that the decoder could learn to detect instead of learning rotation
        robustness).
        """
        max_angle = 30.0 * strength
        angle_deg = random.uniform(-max_angle, max_angle)
        angle_rad = angle_deg * math.pi / 180.0

        B = image.size(0)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        # 2×3 affine matrix for rotation (same for all images in batch)
        theta = torch.tensor([
            [cos_a, -sin_a, 0.0],
            [sin_a,  cos_a, 0.0],
        ], dtype=image.dtype, device=image.device).unsqueeze(0).expand(B, -1, -1)

        grid = F.affine_grid(theta, image.size(), align_corners=False)
        rotated = F.grid_sample(image, grid, align_corners=False,
                                mode='bilinear', padding_mode='border')
        return rotated

    def _avg_blur(self, image: torch.Tensor, strength: float) -> torch.Tensor:
        """
        Branch 6: Average (box) blur.

        Kernel size sampled from {3, 5, 7} (odd only). Implemented via
        depthwise average convolution with uniform weights.
        """
        # Choose kernel size; at low strength, only use k=3
        max_k = int(3 + 4 * strength)  # strength=0.5→5, strength=1.0→7
        # Ensure odd
        if max_k % 2 == 0:
            max_k -= 1
        max_k = max(3, max_k)

        k = random.choice([k for k in range(3, max_k + 1, 2)])  # odd sizes only
        padding = k // 2

        # Depthwise convolution: each channel blurred independently
        C = image.size(1)
        kernel = torch.ones(C, 1, k, k, device=image.device, dtype=image.dtype) / (k * k)
        return F.conv2d(image, kernel, padding=padding, groups=C)

    def _screenshot(self, image: torch.Tensor, strength: float) -> torch.Tensor:
        """
        Branch 7: Screenshotting simulation.

        Sequential composition that mimics the degradation pipeline of
        taking a screenshot of a watermarked image displayed on screen:
            1. JPEG compression (q=70-85) — display/capture compression
            2. Gaussian blur (sigma=0.3-0.8) — display pixel blending
            3. Brightness/contrast jitter (±5%) — display calibration variance
        """
        # ── JPEG ────────────────────────────────────────────────────────────────
        jpeg_q = random.uniform(70, 85)
        x = self.diff_jpeg(image, quality=jpeg_q)

        # ── Gaussian blur ───────────────────────────────────────────────────────
        sigma = max(0.5, random.uniform(0.3, 0.8)) * strength
        # Create Gaussian kernel
        k_size = 5  # Fixed small kernel for efficiency
        ax = torch.arange(k_size, device=image.device, dtype=image.dtype) - k_size // 2
        gauss_1d = torch.exp(-0.5 * (ax / sigma) ** 2)
        gauss_1d = gauss_1d / gauss_1d.sum()
        gauss_2d = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)  # outer product
        gauss_2d = gauss_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, k, k)

        C = x.size(1)
        gauss_kernel = gauss_2d.expand(C, 1, k_size, k_size)
        x = F.conv2d(x, gauss_kernel, padding=k_size // 2, groups=C)

        # ── Brightness/contrast jitter ──────────────────────────────────────────
        brightness = 1.0 + random.uniform(-0.05, 0.05) * strength
        contrast = 1.0 + random.uniform(-0.05, 0.05) * strength
        mean = x.mean(dim=[2, 3], keepdim=True)
        x = contrast * (x - mean) + mean + (brightness - 1.0)
        x = x.clamp(0.0, 1.0)

        return x

    def forward(
        self,
        image: torch.Tensor,
        current_epoch: int = 1,
    ) -> List[torch.Tensor]:
        """
        Apply all active attack branches simultaneously.

        Args:
            image:         (B, 3, 256, 256) watermarked image in [0, 1].
            current_epoch: Current training epoch (1-indexed) for curriculum.

        Returns:
            List of (B, 3, 256, 256) attacked images, one per active branch.
            The training loop averages loss across all branches.
        """
        active_branches = self._get_active_branches(current_epoch)
        strength = self._get_strength_multiplier(current_epoch)

        # Dispatch table for branch execution
        branch_fns = {
            0: lambda img: self._identity(img),
            1: lambda img: self._jpeg(img, strength),
            2: lambda img: self._gaussian_noise(img, strength),
            3: lambda img: self._salt_pepper(img, strength),
            4: lambda img: self._pixel_dropout(img, strength),
            5: lambda img: self._rotation(img, strength),
            6: lambda img: self._avg_blur(img, strength),
            7: lambda img: self._screenshot(img, strength),
        }

        attacked_images = []
        for branch_id in active_branches:
            attacked = branch_fns[branch_id](image)
            attacked_images.append(attacked)

        return attacked_images

    def get_active_branch_names(self, epoch: int) -> List[str]:
        """Get human-readable names of active branches at given epoch."""
        indices = self._get_active_branches(epoch)
        return [self.branch_names[i] for i in indices]

    def apply_single_attack(
        self,
        image: torch.Tensor,
        attack_name: str,
        strength: float = 1.0,
    ) -> torch.Tensor:
        """
        Apply a single named attack (for evaluation).

        Args:
            image:       (B, 3, H, W) image in [0, 1].
            attack_name: One of self.branch_names.
            strength:    Attack strength multiplier.

        Returns:
            (B, 3, H, W) attacked image.
        """
        dispatch = {
            "identity": lambda: self._identity(image),
            "jpeg": lambda: self._jpeg(image, strength),
            "gaussian_noise": lambda: self._gaussian_noise(image, strength),
            "salt_pepper": lambda: self._salt_pepper(image, strength),
            "pixel_dropout": lambda: self._pixel_dropout(image, strength),
            "rotation": lambda: self._rotation(image, strength),
            "avg_blur": lambda: self._avg_blur(image, strength),
            "screenshot": lambda: self._screenshot(image, strength),
        }
        return dispatch[attack_name]()
