"""
Full training loop for the PyTorch watermarking system.

Trains the encoder-decoder pair end-to-end against the multi-branch
attack simulator with curriculum scheduling.

Usage:
    python trainer.py --data-dir ./train_images --epochs 100
    python trainer.py --data-dir ./train_images --epochs 5  # smoke test
"""

import os
import sys
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Optional

from config import get_config, Config
from models.encoder import WatermarkEncoder
from models.decoder import WatermarkDecoder
from attacks.simulator import AttackSimulator
from losses import WatermarkLoss
from data.loader import get_dataloader


def compute_psnr(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """
    Compute Peak Signal-to-Noise Ratio between two images.

    Args:
        original:      (B, C, H, W) in [0, 1].
        reconstructed: (B, C, H, W) in [0, 1].

    Returns:
        PSNR in dB (averaged over batch). Higher is better.
    """
    mse = torch.mean((original - reconstructed) ** 2).item()
    if mse < 1e-10:
        return 100.0  # Effectively identical
    return 10.0 * np.log10(1.0 / mse)


def compute_ber(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute Bit Error Rate between predicted and target watermarks.

    Args:
        predictions: (B, 256) logits (pre-sigmoid).
        targets:     (B, 256) binary targets.

    Returns:
        BER as a fraction in [0, 1]. Lower is better. 0.5 = random chance.
    """
    predicted_bits = (torch.sigmoid(predictions) > 0.5).float()
    errors = (predicted_bits != targets).float()
    return errors.mean().item()


def compute_bit_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute bit accuracy (1 - BER) as a percentage."""
    return (1.0 - compute_ber(predictions, targets)) * 100.0


class Trainer:
    """
    Orchestrates the full training loop with curriculum scheduling.

    Features:
        - Multi-branch attack simulation (all branches simultaneously)
        - Curriculum-based attack strength progression
        - Gradient clipping for stability
        - Best-loss and best-PSNR checkpoint saving
        - Per-epoch PSNR, BER, and loss component logging
        - Optional mixed precision for Kaggle mode
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # ── Models ──────────────────────────────────────────────────────────────
        self.encoder = WatermarkEncoder(
            watermark_length=cfg.watermark_length,
            num_blocks=cfg.encoder_residual_blocks,
            filters=cfg.encoder_filters,
            delta_scale=cfg.delta_scale,
            ycbcr_gain=cfg.ycbcr_gain,
            use_checkpointing=cfg.use_gradient_checkpointing,
        ).to(self.device)

        self.decoder = WatermarkDecoder(
            watermark_length=cfg.watermark_length,
            num_blocks=cfg.decoder_residual_blocks,
            filters=cfg.decoder_filters,
            attention_reduction=cfg.attention_reduction,
            use_checkpointing=cfg.use_gradient_checkpointing,
        ).to(self.device)

        # ── Attack simulator ────────────────────────────────────────────────────
        self.attack_sim = AttackSimulator(
            curriculum_phase1_end=cfg.curriculum_phase1_end,
            curriculum_phase2_end=cfg.curriculum_phase2_end,
            curriculum_phase3_end=cfg.curriculum_phase3_end,
        ).to(self.device)

        # ── Loss function ───────────────────────────────────────────────────────
        self.criterion = WatermarkLoss(
            lambda_mse=cfg.lambda_mse,
            lambda_lpips=cfg.lambda_lpips,
            lambda_ssim=cfg.lambda_ssim,
            lambda_bce=cfg.lambda_bce,
            lambda_delta=cfg.lambda_delta,
            label_smooth_pos=cfg.label_smooth_pos,
            label_smooth_neg=cfg.label_smooth_neg,
            device=cfg.device,
        ).to(self.device)

        # ── Optimizer ───────────────────────────────────────────────────────────
        # Only encoder and decoder params are trainable — attack simulator and
        # LPIPS network have no trainable parameters.
        trainable_params = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
        )
        self.optimizer = optim.Adam(
            trainable_params,
            lr=cfg.learning_rate,
        )

        # ── LR scheduler ───────────────────────────────────────────────────────
        # Halve LR when loss plateaus for 5 epochs, floor at 1e-7
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5,
            min_lr=1e-7,
        )

        # ── Mixed precision scaler (Kaggle mode only) ──────────────────────────
        self.scaler = torch.amp.GradScaler('cuda') if cfg.use_amp else None

        # ── Checkpointing ───────────────────────────────────────────────────────
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        self.best_loss = float('inf')
        self.best_psnr = 0.0

    def _save_checkpoint(self, epoch: int, tag: str) -> None:
        """Save encoder + decoder weights with a descriptive tag."""
        path = os.path.join(self.cfg.checkpoint_dir, f"best_{tag}.pt")
        torch.save({
            'epoch': epoch,
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'best_psnr': self.best_psnr,
        }, path)

    def load_checkpoint(self, path: str) -> int:
        """
        Load a checkpoint and return the epoch it was saved at.

        Args:
            path: Path to the .pt checkpoint file.

        Returns:
            Epoch number the checkpoint was saved at.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.best_psnr = checkpoint.get('best_psnr', 0.0)
        return checkpoint.get('epoch', 0)

    def train_one_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        The key MBRS insight: all active attack branches run simultaneously
        on each batch, and the decoder's loss is averaged across all branches.
        This forces the encoder/decoder to be jointly robust to all attacks.

        Returns:
            Dict of averaged metrics for the epoch.
        """
        self.encoder.train()
        self.decoder.train()

        # Accumulators
        total_loss_sum = 0.0
        psnr_sum = 0.0
        ber_sum = 0.0
        component_sums: Dict[str, float] = {}
        num_batches = 0

        active_attacks = self.attack_sim.get_active_branch_names(epoch)

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=True)
        for images, watermarks in pbar:
            images = images.to(self.device, non_blocking=True)
            watermarks = watermarks.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # ── Forward: Encode ─────────────────────────────────────────────
            if self.cfg.use_amp:
                with torch.amp.autocast('cuda'):
                    watermarked, delta = self.encoder(images, watermarks)
            else:
                watermarked, delta = self.encoder(images, watermarks)

            # ── Forward: Attack (all branches simultaneously) ───────────────
            attacked_images = self.attack_sim(watermarked, current_epoch=epoch)

            # ── Forward: Decode each branch + aggregate loss ────────────────
            total_branch_loss = torch.tensor(0.0, device=self.device)
            branch_ber_sum = 0.0
            num_branches = len(attacked_images)

            for attacked_img in attacked_images:
                if self.cfg.use_amp:
                    with torch.amp.autocast('cuda'):
                        logits = self.decoder(attacked_img)
                        loss, components = self.criterion(
                            images, watermarked, logits, watermarks, delta,
                            epoch=epoch,
                        )
                else:
                    logits = self.decoder(attacked_img)
                    loss, components = self.criterion(
                        images, watermarked, logits, watermarks, delta,
                        epoch=epoch,
                    )

                total_branch_loss = total_branch_loss + loss
                branch_ber_sum += compute_ber(logits, watermarks)

                # Accumulate component stats
                for k, v in components.items():
                    component_sums[k] = component_sums.get(k, 0.0) + v

            # Average loss across branches
            avg_loss = total_branch_loss / num_branches

            # ── Backward + step ─────────────────────────────────────────────
            if self.scaler is not None:
                self.scaler.scale(avg_loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.decoder.parameters()),
                    self.cfg.grad_clip_norm,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                avg_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.decoder.parameters()),
                    self.cfg.grad_clip_norm,
                )
                self.optimizer.step()

            # ── Metrics ─────────────────────────────────────────────────────
            psnr = compute_psnr(images, watermarked)
            avg_ber = branch_ber_sum / num_branches

            total_loss_sum += avg_loss.item()
            psnr_sum += psnr
            ber_sum += avg_ber
            num_batches += 1

            pbar.set_postfix({
                'loss': f'{avg_loss.item():.4f}',
                'psnr': f'{psnr:.1f}',
                'ber': f'{avg_ber:.3f}',
            })

        # ── Epoch averages ──────────────────────────────────────────────────────
        n = max(num_batches, 1)
        nb = n * num_branches  # total branch evaluations for component averaging
        metrics = {
            'loss': total_loss_sum / n,
            'psnr': psnr_sum / n,
            'ber': ber_sum / n,
            'bit_acc': (1.0 - ber_sum / n) * 100.0,
        }
        for k, v in component_sums.items():
            metrics[f'loss_{k}'] = v / nb

        return metrics

    def train(
        self,
        dataloader: torch.utils.data.DataLoader,
        start_epoch: int = 1,
    ) -> None:
        """
        Full training loop with curriculum scheduling and checkpointing.

        Args:
            dataloader:  Training data loader.
            start_epoch: Epoch to start from (for resuming).
        """
        print(f"\n{'='*70}")
        print(f"  PyTorch Watermarking System — Training")
        print(f"  Device: {self.device}")
        print(f"  Batch size: {self.cfg.batch_size}")
        print(f"  Mixed precision: {self.cfg.use_amp}")
        print(f"  Gradient checkpointing: {self.cfg.use_gradient_checkpointing}")
        print(f"  Epochs: {start_epoch} → {self.cfg.epochs}")
        print(f"{'='*70}\n")

        for epoch in range(start_epoch, self.cfg.epochs + 1):
            active = self.attack_sim.get_active_branch_names(epoch)
            print(f"\n── Epoch {epoch}/{self.cfg.epochs} "
                  f"[Active attacks: {', '.join(active)}] ──")

            metrics = self.train_one_epoch(dataloader, epoch)

            # LR scheduler step
            self.scheduler.step(metrics['loss'])

            # ── Epoch summary ───────────────────────────────────────────────
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"  Loss: {metrics['loss']:.4f}  |  "
                  f"PSNR: {metrics['psnr']:.2f} dB  |  "
                  f"BER: {metrics['ber']:.4f}  |  "
                  f"Bit Acc: {metrics['bit_acc']:.1f}%  |  "
                  f"LR: {current_lr:.2e}")
            print(f"  Components — "
                  f"BCE: {metrics.get('loss_bce', 0):.4f}  "
                  f"EmbedRMS: {metrics.get('loss_embed_rms', 0):.5f}  "
                  f"MSE: {metrics.get('loss_mse', 0):.4f}  "
                  f"LPIPS: {metrics.get('loss_lpips', 0):.4f}  "
                  f"SSIM: {metrics.get('loss_ssim', 0):.4f}  "
                  f"Delta: {metrics.get('loss_delta_l2', 0):.4f}")

            # ── Checkpointing ───────────────────────────────────────────────
            if metrics['loss'] < self.best_loss:
                self.best_loss = metrics['loss']
                self._save_checkpoint(epoch, 'loss')
                print(f"  ✓ New best loss: {self.best_loss:.4f}")

            if metrics['psnr'] > self.best_psnr:
                self.best_psnr = metrics['psnr']
                self._save_checkpoint(epoch, 'psnr')
                print(f"  ✓ New best PSNR: {self.best_psnr:.2f} dB")

        print(f"\n{'='*70}")
        print(f"  Training complete!")
        print(f"  Best loss: {self.best_loss:.4f}")
        print(f"  Best PSNR: {self.best_psnr:.2f} dB")
        print(f"{'='*70}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train watermarking system")
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to training images directory')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size override')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate override')
    args = parser.parse_args()

    cfg = get_config()

    # Apply CLI overrides
    if args.data_dir:
        cfg.train_data_dir = args.data_dir
    if args.epochs:
        cfg.epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.lr:
        cfg.learning_rate = args.lr

    # ── DataLoader ──────────────────────────────────────────────────────────────
    dataloader = get_dataloader(
        image_dir=cfg.train_data_dir,
        batch_size=cfg.batch_size,
        image_size=cfg.image_size,
        watermark_length=cfg.watermark_length,
        num_workers=cfg.num_workers,
        augment=True,
        shuffle=True,
    )
    print(f"Dataset: {len(dataloader.dataset)} images, "
          f"{len(dataloader)} batches/epoch")

    # ── Train ───────────────────────────────────────────────────────────────────
    trainer = Trainer(cfg)

    start_epoch = 1
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume) + 1
        print(f"Resumed from checkpoint, starting at epoch {start_epoch}")

    trainer.train(dataloader, start_epoch=start_epoch)


if __name__ == '__main__':
    main()
