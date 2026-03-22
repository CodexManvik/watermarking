"""
DWT-based watermark encoder.

Pipeline:
    RGB → YCbCr → DWT (Haar, 1-level) → extract LL subbands →
    concatenate with upsampled watermark → residual embedding CNN →
    adaptive-scale delta perturbation → add to LL → IDWT → YCbCr → RGB

Uses pytorch_wavelets for the DWT/IDWT.
Install: pip install pytorch-wavelets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from .haar_dwt import HaarDWT, HaarIDWT
from typing import Tuple


# ── Colorspace conversion matrices ─────────────────────────────────────────────
# ITU-R BT.601 conversion.  Kept as module-level constants so they're only
# constructed once and moved to device via .to() when needed.

# RGB → YCbCr offsets: Y has no offset, Cb/Cr are shifted by 0.5 to center at 0
_YCBCR_OFFSET = torch.tensor([0.0, 0.5, 0.5]).reshape(1, 3, 1, 1)

# Forward matrix: each row produces one of Y, Cb, Cr from R, G, B
_RGB_TO_YCBCR = torch.tensor([
    [ 0.299,     0.587,     0.114   ],
    [-0.168736, -0.331264,  0.5     ],
    [ 0.5,      -0.418688, -0.081312],
]).float()

# Inverse matrix: each row produces one of R, G, B from Y, Cb-0.5, Cr-0.5
_YCBCR_TO_RGB = torch.tensor([
    [1.0,  0.0,       1.402  ],
    [1.0, -0.344136, -0.714136],
    [1.0,  1.772,     0.0    ],
]).float()


def rgb_to_ycbcr(rgb: torch.Tensor) -> torch.Tensor:
    """Convert (B, 3, H, W) RGB [0,1] to YCbCr. Y in [0,1], Cb/Cr in [0,1]."""
    # Einstein summation: oc = output channel, ic = input channel
    # The matmul is done per-pixel across channels.
    weight = _RGB_TO_YCBCR.to(rgb.device)
    ycbcr = torch.einsum('oc,bchw->bohw', weight, rgb)
    ycbcr = ycbcr + _YCBCR_OFFSET.to(rgb.device)
    return ycbcr


def ycbcr_to_rgb(ycbcr: torch.Tensor) -> torch.Tensor:
    """Convert (B, 3, H, W) YCbCr back to RGB [0,1]."""
    weight = _YCBCR_TO_RGB.to(ycbcr.device)
    shifted = ycbcr - _YCBCR_OFFSET.to(ycbcr.device)
    rgb = torch.einsum('oc,bchw->bohw', weight, shifted)
    return rgb.clamp(0.0, 1.0)


class ResidualBlock(nn.Module):
    """
    Pre-activation residual block (BN → ReLU → Conv → BN → ReLU → Conv + skip).

    Pre-activation ordering gives better gradient flow than the original
    post-activation design (He et al., 2016), which matters here because the
    embedding CNN only has 3 blocks and every gradient counts.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class WatermarkUpsampler(nn.Module):
    """
    Upsample a flat watermark vector to a 2D spatial map via transposed convolutions.

    Path: (B, 256) → view (B, 1, 16, 16) → ConvTranspose2d chain → (B, 1, 128, 128)

    Using view (not FC) preserves the spatial structure of the reshaped vector,
    and transposed convolutions learn how to spatially spread each bit's influence
    over a local neighborhood — critical for robust spatial learning.
    """

    def __init__(self, watermark_length: int = 256) -> None:
        super().__init__()
        # 256 = 16 * 16 * 1, so we reshape to (B, 1, 16, 16) first via view.
        # Then three transposed conv layers double spatial dims each time:
        #   16→32, 32→64, 64→128
        self.reshape_size = (1, 16, 16)  # must multiply to watermark_length

        self.upsample = nn.Sequential(
            # 16×16 → 32×32
            nn.ConvTranspose2d(1, 64, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 32×32 → 64×64
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 64×64 → 128×128
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            # No final activation — the concatenation layer handles normalization
        )

    def forward(self, watermark: torch.Tensor) -> torch.Tensor:
        """
        Args:
            watermark: (B, 256) binary watermark vector.
        Returns:
            (B, 1, 128, 128) spatial watermark map.
        """
        B = watermark.size(0)
        # view, NOT a learned FC layer — preserves spatial structure
        x = watermark.view(B, *self.reshape_size)  # (B, 1, 16, 16)
        x = self.upsample(x)                        # (B, 1, 128, 128)
        return x


class EmbeddingCNN(nn.Module):
    """
    Residual CNN that produces the delta perturbation from concatenated
    LL band + watermark map.

    Input:  (B, 2, 128, 128)  — [LL_band, watermark_map] concatenated
    Output: (B, 1, 128, 128)  — delta bounded to [-1, 1] via tanh
    """

    def __init__(self, num_blocks: int = 3, filters: int = 64,
                 use_checkpointing: bool = False) -> None:
        super().__init__()
        self.use_checkpointing = use_checkpointing

        # Entry conv: expand from 2 channels (LL + watermark) to `filters`
        self.entry = nn.Sequential(
            nn.Conv2d(2, filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(filters) for _ in range(num_blocks)]
        )

        # Exit conv: squeeze back to 1 channel with tanh activation
        self.exit = nn.Sequential(
            nn.Conv2d(filters, 1, 3, padding=1),
            nn.Tanh(),  # Bounds output to [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.entry(x)

        for block in self.res_blocks:
            if self.use_checkpointing and self.training:
                # Trade compute for VRAM — recompute block activations during
                # backward pass instead of storing them.
                x = grad_checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        return self.exit(x)


class WatermarkEncoder(nn.Module):
    """
    Full encoder: embeds a 256-bit watermark into an RGB image via DWT.

    Steps:
        1. RGB → YCbCr
        2. DWT per channel → LL subbands (128×128 each)
        3. Upsample watermark → 128×128 spatial map
        4. Concatenate watermark map with Y-channel LL band
        5. Residual CNN → delta perturbation (tanh-bounded)
        6. Texture-adaptive scaling (gradient magnitude * delta)
        7. Apply delta to LL bands with YCbCr 4:1:1 gain
        8. IDWT → YCbCr → RGB
    """

    def __init__(
        self,
        watermark_length: int = 256,
        num_blocks: int = 3,
        filters: int = 64,
        delta_scale: float = 0.55,
        ycbcr_gain: Tuple[float, float, float] = (1.0, 0.25, 0.25),
        use_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.delta_scale = delta_scale
        # Register gain as a buffer so it moves to the correct device with .to()
        self.register_buffer(
            'ycbcr_gain',
            torch.tensor(ycbcr_gain).view(1, 3, 1, 1)
        )

        # Pure-PyTorch Haar DWT / IDWT — no external dependency.
        # Single-level: LL output is (B, C, H//2, W//2).
        self.dwt = HaarDWT()
        self.idwt = HaarIDWT()

        self.upsampler = WatermarkUpsampler(watermark_length)

        self.embedding_cnn = EmbeddingCNN(
            num_blocks=num_blocks,
            filters=filters,
            use_checkpointing=use_checkpointing,
        )

    def _compute_texture_map(self, image: torch.Tensor) -> torch.Tensor:
        """
        Compute per-pixel texture complexity as gradient magnitude.

        High-gradient regions (edges, textures) can tolerate larger perturbations
        without visible artifacts, so we scale the delta proportionally.

        Args:
            image: (B, C, H, W) input image in any colorspace.
        Returns:
            (B, 1, H, W) normalized gradient magnitude map in [0, 1].
        """
        # Convert to grayscale for gradient computation
        gray = image.mean(dim=1, keepdim=True)  # (B, 1, H, W)

        # Sobel-like gradients via finite differences
        # Pad to maintain spatial dimensions
        gx = gray[:, :, :, 1:] - gray[:, :, :, :-1]  # horizontal gradient
        gy = gray[:, :, 1:, :] - gray[:, :, :-1, :]   # vertical gradient

        # Pad back to original size (right/bottom edge)
        gx = F.pad(gx, (0, 1, 0, 0))  # pad width dimension
        gy = F.pad(gy, (0, 0, 0, 1))  # pad height dimension

        # Gradient magnitude
        magnitude = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

        # Normalize to [0, 1] per image, then add a floor of 0.3 so that
        # even flat regions get some watermark embedding (otherwise the
        # watermark would only exist in textured areas and be easy to attack).
        mag_max = magnitude.flatten(1).max(dim=1)[0].view(-1, 1, 1, 1) + 1e-8
        magnitude = magnitude / mag_max
        magnitude = 0.3 + 0.7 * magnitude  # floor at 0.3, ceiling at 1.0

        return magnitude

    def forward(
        self,
        image: torch.Tensor,
        watermark: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embed watermark into image.

        Args:
            image:     (B, 3, 256, 256) RGB image in [0, 1].
            watermark: (B, 256) binary watermark vector (0s and 1s).

        Returns:
            watermarked_image: (B, 3, 256, 256) RGB watermarked image in [0, 1].
            delta:             (B, 3, 128, 128) embedding perturbation (for loss).
        """
        B = image.size(0)

        # ── Step 1: RGB → YCbCr ────────────────────────────────────────────────
        ycbcr = rgb_to_ycbcr(image)  # (B, 3, 256, 256)

        # ── Step 2: DWT per channel ─────────────────────────────────────────────
        # HaarDWT returns:
        #   yl: (B, C, H//2, W//2) — LL subband
        #   yh: (B, C, 3, H//2, W//2) — [LH, HL, HH] packed
        yl, yh = self.dwt(ycbcr)  # yl: (B, 3, 128, 128)

        # Extract Y channel LL band for primary embedding
        y_ll = yl[:, 0:1, :, :]  # (B, 1, 128, 128)

        # ── Step 3: Upsample watermark ──────────────────────────────────────────
        wm_spatial = self.upsampler(watermark)  # (B, 1, 128, 128)

        # ── Step 4: Concatenate and embed ───────────────────────────────────────
        concat = torch.cat([y_ll, wm_spatial], dim=1)  # (B, 2, 128, 128)

        # ── Step 5: Delta perturbation ──────────────────────────────────────────
        raw_delta = self.embedding_cnn(concat)  # (B, 1, 128, 128), tanh-bounded

        # ── Step 6: Texture-adaptive scaling ────────────────────────────────────
        # Compute texture map at LL resolution (downsample image first)
        image_ll_res = F.interpolate(image, size=(128, 128), mode='bilinear',
                                     align_corners=False)
        texture_map = self._compute_texture_map(image_ll_res)  # (B, 1, 128, 128)

        # Scale delta: tanh * delta_scale * texture_complexity
        scaled_delta = raw_delta * self.delta_scale * texture_map  # (B, 1, 128, 128)

        # ── Step 7: Apply delta with YCbCr 4:1:1 gain ──────────────────────────
        # Expand delta to all 3 channels, multiply by gain:
        #   Y  gets full delta   (gain = 1.0)
        #   Cb gets 1/4 delta    (gain = 0.25)
        #   Cr gets 1/4 delta    (gain = 0.25)
        # This is the explicit gain multiplication the user flagged — NOT a comment.
        delta_3ch = scaled_delta.expand(B, 3, 128, 128) * self.ycbcr_gain  # (B, 3, 128, 128)

        # Add delta to LL subband
        new_yl = yl + delta_3ch  # (B, 3, 128, 128)

        # ── Step 8: IDWT → YCbCr → RGB ─────────────────────────────────────────
        reconstructed_ycbcr = self.idwt(new_yl, yh)  # (B, 3, 256, 256)
        watermarked_rgb = ycbcr_to_rgb(reconstructed_ycbcr)  # (B, 3, 256, 256)

        return watermarked_rgb, delta_3ch
