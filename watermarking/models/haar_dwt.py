"""
Pure-PyTorch Haar Discrete Wavelet Transform (DWT) and its inverse (IDWT).

Replaces pytorch_wavelets to avoid the pkg_resources / Python 3.12 incompatibility.

The Haar wavelet filters are fixed (not learned):
    Low-pass  h_lo = [1, 1] / sqrt(2)
    High-pass h_hi = [1,-1] / sqrt(2)

A 2D single-level DWT is applied as two sequential 1D convolutions:
    1. Along rows   → L and H row-filtered
    2. Along columns → LL, LH, HL, HH subbands

Each subband is half the spatial resolution of the input.

Usage:
    dwt  = HaarDWT()
    idwt = HaarIDWT()
    yl, yh = dwt(x)        # yl: (B,C,H/2,W/2) LL subband
                            # yh: (B,C,3,H/2,W/2) [LH, HL, HH] subbands
    x_rec = idwt(yl, yh)   # (B,C,H,W) reconstructed signal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def _build_haar_filters(device: torch.device,
                        dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build the 1D Haar low-pass and high-pass filters as 2D convolution kernels.

    The filters are scaled by 1/sqrt(2) (orthonormal Haar convention) so that
    the DWT is energy-preserving: the L2 norm of the output equals the L2 norm
    of the input, which keeps the LL band in the same numerical range as the
    original — important for the delta perturbation scaling to be predictable.

    The kernels are shaped (1, 1, 1, 2) for F.conv2d horizontal application and
    (1, 1, 2, 1) for vertical application.
    """
    s = 2 ** -0.5  # 1/sqrt(2)
    lo = torch.tensor([[[[s, s]]]], dtype=dtype, device=device)   # (1,1,1,2) low
    hi = torch.tensor([[[[s, -s]]]], dtype=dtype, device=device)  # (1,1,1,2) high
    return lo, hi


class HaarDWT(nn.Module):
    """
    Single-level 2D Haar DWT returning LL, LH, HL, HH subbands.

    Returns the same interface as pytorch_wavelets DWTForward(J=1):
        yl:  (B, C, H//2, W//2)        — LL subband
        yh:  (B, C, 3, H//2, W//2)     — [LH, HL, HH] packed as one tensor

    The subband ordering (LH, HL, HH) matches the pytorch_wavelets convention.
    """

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, H, W) input in any value range.

        Returns:
            yl: (B, C, H//2, W//2) LL subband.
            yh: (B, C, 3, H//2, W//2) — dim-2 contains [LH, HL, HH].
        """
        B, C, H, W = x.shape
        lo, hi = _build_haar_filters(x.device, x.dtype)

        # Reshape to treat all B*C as independent channels for group conv
        # This avoids an explicit loop over channels.
        xr = x.reshape(B * C, 1, H, W)

        # ── Step 1: Filter along width (horizontal) ─────────────────────────
        # Each filter is (out_ch, in_ch/groups, kH, kW) = (1, 1, 1, 2)
        # Pad right by 1 so we get exact W//2 outputs with stride 2
        xr_pad = F.pad(xr, (0, 1, 0, 0), mode='reflect')  # pad right
        row_lo = F.conv2d(xr_pad, lo, stride=(1, 2))  # (B*C, 1, H, W//2)
        row_hi = F.conv2d(xr_pad, hi, stride=(1, 2))  # (B*C, 1, H, W//2)

        # ── Step 2: Filter along height (vertical) ──────────────────────────
        lo_v = lo.transpose(2, 3)  # (1,1,2,1)
        hi_v = hi.transpose(2, 3)

        def vconv(inp: torch.Tensor, kern: torch.Tensor) -> torch.Tensor:
            inp_pad = F.pad(inp, (0, 0, 0, 1), mode='reflect')  # pad bottom
            return F.conv2d(inp_pad, kern, stride=(2, 1))

        LL = vconv(row_lo, lo_v)   # (B*C, 1, H//2, W//2)
        LH = vconv(row_lo, hi_v)   # horizontal low, vertical high
        HL = vconv(row_hi, lo_v)   # horizontal high, vertical low
        HH = vconv(row_hi, hi_v)   # horizontal high, vertical high

        # Reshape back to (B, C, H//2, W//2)
        Hh, Wh = H // 2, W // 2
        yl = LL.reshape(B, C, Hh, Wh)

        # Pack LH, HL, HH into yh: (B, C, 3, Hh, Wh)
        yh = torch.stack([
            LH.reshape(B, C, Hh, Wh),
            HL.reshape(B, C, Hh, Wh),
            HH.reshape(B, C, Hh, Wh),
        ], dim=2)

        return yl, yh


class HaarIDWT(nn.Module):
    """
    Single-level 2D inverse Haar DWT.

    Takes the same (yl, yh) format produced by HaarDWT and reconstructs the
    full-resolution signal.

    Reconstruction uses transpose convolutions (ConvTranspose2d equivalent via
    F.conv_transpose2d) with the same Haar filters.
    """

    def forward(
        self,
        yl: torch.Tensor,
        yh: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            yl: (B, C, H//2, W//2) LL subband.
            yh: (B, C, 3, H//2, W//2) [LH, HL, HH] subbands.

        Returns:
            (B, C, H, W) reconstructed signal.
        """
        B, C, Hh, Wh = yl.shape
        H, W = Hh * 2, Wh * 2

        LH = yh[:, :, 0, :, :]  # (B, C, Hh, Wh)
        HL = yh[:, :, 1, :, :]
        HH = yh[:, :, 2, :, :]

        lo, hi = _build_haar_filters(yl.device, yl.dtype)
        lo_v = lo.transpose(2, 3)
        hi_v = hi.transpose(2, 3)

        def itconv_h(inp: torch.Tensor, kern: torch.Tensor) -> torch.Tensor:
            """Inverse 1D convolution along width (upsample × 2 horizontally)."""
            # conv_transpose2d with stride 2 inserts zeros between samples
            out = F.conv_transpose2d(inp.reshape(-1, 1, inp.size(-2), inp.size(-1)),
                                     kern, stride=(1, 2))
            # Remove the extra column added by transpose conv
            return out[:, :, :, :W].reshape(inp.shape[0], inp.shape[1],
                                             inp.shape[2], W)

        def itconv_v(inp: torch.Tensor, kern: torch.Tensor) -> torch.Tensor:
            """Inverse 1D convolution along height (upsample × 2 vertically)."""
            out = F.conv_transpose2d(inp.reshape(-1, 1, inp.shape[-2], inp.shape[-1]),
                                     kern, stride=(2, 1))
            return out[:, :, :H, :].reshape(inp.shape[0], inp.shape[1], H,
                                             inp.shape[3])

        # ── Reconstruct row-filtered signals from column subbands ────────────
        # LL + LH → row_lo reconstructed
        LL_up = itconv_v(yl.reshape(B * C, 1, Hh, Wh)
                          .reshape(B, C, Hh, Wh), lo_v)
        LH_up = itconv_v(LH, hi_v)

        HL_up = itconv_v(HL, lo_v)
        HH_up = itconv_v(HH, hi_v)

        row_lo = LL_up + LH_up   # (B, C, H, Wh)
        row_hi = HL_up + HH_up

        # ── Reconstruct full image from row signals ───────────────────────────
        lo_col_up = itconv_h(row_lo, lo)
        hi_col_up = itconv_h(row_hi, hi)

        return lo_col_up + hi_col_up  # (B, C, H, W)
