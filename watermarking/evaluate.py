"""
Evaluation script for the watermarking system.

Runs each attack individually at fixed strength and prints a per-attack
results table with PSNR, SSIM, BER, and Bit Accuracy.

Usage:
    python evaluate.py --checkpoint checkpoints/best_loss.pt --data-dir test_images
    python evaluate.py --checkpoint checkpoints/best_loss.pt --data-dir test_images --num-images 50
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from pytorch_msssim import ssim
from typing import Dict, List, Tuple

from config import get_config, Config
from models.encoder import WatermarkEncoder
from models.decoder import WatermarkDecoder
from attacks.simulator import AttackSimulator
from data.loader import get_dataloader


def compute_psnr(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """PSNR in dB between two image tensors."""
    mse = torch.mean((original - reconstructed) ** 2).item()
    if mse < 1e-10:
        return 100.0
    return 10.0 * np.log10(1.0 / mse)


def compute_ber(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Bit Error Rate: fraction of incorrectly recovered bits."""
    predicted_bits = (torch.sigmoid(predictions) > 0.5).float()
    return (predicted_bits != targets).float().mean().item()


def evaluate_attack(
    encoder: WatermarkEncoder,
    decoder: WatermarkDecoder,
    attack_sim: AttackSimulator,
    dataloader: torch.utils.data.DataLoader,
    attack_name: str,
    device: torch.device,
    num_images: int = -1,
) -> Dict[str, float]:
    """
    Evaluate one specific attack on the full dataset (or subset).

    Returns dict with averaged PSNR, SSIM, BER, and bit accuracy.
    """
    psnr_vals: List[float] = []
    ssim_vals: List[float] = []
    ber_vals: List[float] = []
    processed = 0

    for images, watermarks in dataloader:
        if 0 < num_images <= processed:
            break

        images = images.to(device)
        watermarks = watermarks.to(device)

        with torch.no_grad():
            # Encode
            watermarked, delta = encoder(images, watermarks)

            # PSNR and SSIM between original and watermarked (before attack)
            psnr = compute_psnr(images, watermarked)
            ssim_val = ssim(watermarked, images, data_range=1.0,
                           size_average=True).item()

            # Attack
            if attack_name == "identity":
                attacked = watermarked
            else:
                attacked = attack_sim.apply_single_attack(
                    watermarked, attack_name, strength=1.0
                )

            # Decode
            logits = decoder(attacked)
            ber = compute_ber(logits, watermarks)

        psnr_vals.append(psnr)
        ssim_vals.append(ssim_val)
        ber_vals.append(ber)
        processed += images.size(0)

    return {
        'psnr': np.mean(psnr_vals),
        'ssim': np.mean(ssim_vals),
        'ber': np.mean(ber_vals),
        'bit_acc': (1.0 - np.mean(ber_vals)) * 100.0,
    }


def print_results_table(results: Dict[str, Dict[str, float]]) -> None:
    """Print a formatted per-attack results table."""
    print(f"\n{'='*72}")
    print(f"  Per-Attack Evaluation Results")
    print(f"{'='*72}")
    print(f"  {'Attack':<20} {'PSNR (dB)':>10} {'SSIM':>10} {'BER':>10} {'Bit Acc':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for attack_name, metrics in results.items():
        print(f"  {attack_name:<20} "
              f"{metrics['psnr']:>10.2f} "
              f"{metrics['ssim']:>10.4f} "
              f"{metrics['ber']:>10.4f} "
              f"{metrics['bit_acc']:>9.1f}%")

    print(f"{'='*72}")

    # ── Summary stats ───────────────────────────────────────────────────────
    avg_psnr = np.mean([m['psnr'] for m in results.values()])
    avg_ber = np.mean([m['ber'] for m in results.values()])
    worst_ber = max(m['ber'] for m in results.values())
    worst_attack = max(results.items(), key=lambda x: x[1]['ber'])[0]

    print(f"\n  Average PSNR: {avg_psnr:.2f} dB")
    print(f"  Average BER:  {avg_ber:.4f}")
    print(f"  Worst BER:    {worst_ber:.4f} ({worst_attack})")
    print(f"  {'PASS' if avg_psnr > 30 else 'FAIL'}: PSNR > 30 dB")
    print(f"  {'PASS' if results.get('identity', {}).get('ber', 1.0) < 0.01 else 'FAIL'}: Identity BER < 1%")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate watermarking system")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pt)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to test images directory')
    parser.add_argument('--num-images', type=int, default=-1,
                        help='Max images to evaluate (-1 for all)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Evaluation batch size')
    args = parser.parse_args()

    cfg = get_config()
    device = torch.device(cfg.device)

    if args.data_dir:
        cfg.val_data_dir = args.data_dir

    # ── Load models ─────────────────────────────────────────────────────────────
    encoder = WatermarkEncoder(
        watermark_length=cfg.watermark_length,
        num_blocks=cfg.encoder_residual_blocks,
        filters=cfg.encoder_filters,
        delta_scale=cfg.delta_scale,
        ycbcr_gain=cfg.ycbcr_gain,
    ).to(device)

    decoder = WatermarkDecoder(
        watermark_length=cfg.watermark_length,
        num_blocks=cfg.decoder_residual_blocks,
        filters=cfg.decoder_filters,
        attention_reduction=cfg.attention_reduction,
    ).to(device)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    encoder.eval()
    decoder.eval()
    print(f"  Checkpoint epoch: {checkpoint.get('epoch', '?')}")
    print(f"  Best loss: {checkpoint.get('best_loss', '?')}")

    # ── DataLoader ──────────────────────────────────────────────────────────────
    dataloader = get_dataloader(
        image_dir=cfg.val_data_dir,
        batch_size=args.batch_size,
        image_size=cfg.image_size,
        watermark_length=cfg.watermark_length,
        num_workers=2,
        augment=False,
        shuffle=False,
        drop_last=False,
    )
    print(f"Evaluation dataset: {len(dataloader.dataset)} images")

    # ── Attack simulator ────────────────────────────────────────────────────────
    attack_sim = AttackSimulator().to(device)

    # ── Run evaluation per attack ───────────────────────────────────────────────
    attack_names = attack_sim.branch_names
    results: Dict[str, Dict[str, float]] = {}

    for attack_name in attack_names:
        print(f"  Evaluating: {attack_name}...", end=" ", flush=True)
        metrics = evaluate_attack(
            encoder, decoder, attack_sim, dataloader,
            attack_name, device, args.num_images,
        )
        results[attack_name] = metrics
        print(f"PSNR={metrics['psnr']:.2f}, BER={metrics['ber']:.4f}")

    # ── Print final table ───────────────────────────────────────────────────────
    print_results_table(results)


if __name__ == '__main__':
    main()
