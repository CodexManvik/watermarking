"""
Dataset and DataLoader for the watermarking system.

Loads RGB images from a directory, resizes to 256×256, normalizes to [0,1],
and generates random 256-bit binary watermarks on-the-fly per sample.

The watermarks are generated randomly (not loaded from disk) because:
1. The encoder must work with ANY arbitrary 256-bit watermark.
2. Random watermarks during training prevent overfitting to specific patterns.
3. No need to store/manage watermark files.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import Tuple, Optional
from pathlib import Path


# Supported image formats
_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


class WatermarkDataset(Dataset):
    """
    Image dataset that pairs each image with a random 256-bit watermark.

    Each __getitem__ returns:
        image:     (3, 256, 256) RGB tensor in [0, 1]
        watermark: (256,) binary tensor of 0s and 1s
    """

    def __init__(
        self,
        image_dir: str,
        image_size: Tuple[int, int] = (256, 256),
        watermark_length: int = 256,
        augment: bool = False,
    ) -> None:
        """
        Args:
            image_dir:        Path to directory of images.
            image_size:       Target (H, W) to resize images to.
            watermark_length: Number of bits in the watermark.
            augment:          Whether to apply random augmentations (training only).
        """
        super().__init__()
        self.watermark_length = watermark_length

        # Collect all image paths
        image_dir = Path(image_dir)
        self.image_paths = sorted([
            str(p) for p in image_dir.rglob('*')
            if p.suffix.lower() in _IMAGE_EXTENSIONS
        ])

        if len(self.image_paths) == 0:
            raise FileNotFoundError(
                f"No images found in {image_dir}. "
                f"Supported formats: {_IMAGE_EXTENSIONS}"
            )

        # ── Transform pipeline ──────────────────────────────────────────────────
        # Always resize to target size and convert to tensor (which normalizes to [0,1])
        transform_list = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),  # Handle non-square images
        ]

        if augment:
            # Mild augmentation — we don't want aggressive augmentation because
            # the attack simulator already provides robustness training.
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
            ])

        transform_list.append(transforms.ToTensor())  # → (3, H, W) in [0, 1]

        self.transform = transforms.Compose(transform_list)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            image:     (3, 256, 256) RGB tensor in [0, 1].
            watermark: (256,) binary tensor of 0s and 1s.
        """
        # Load and transform image
        img = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.transform(img)

        # Generate random binary watermark
        # Using randint directly — each bit is independent Bernoulli(0.5)
        watermark = torch.randint(0, 2, (self.watermark_length,), dtype=torch.float32)

        return image, watermark


def get_dataloader(
    image_dir: str,
    batch_size: int = 16,
    image_size: Tuple[int, int] = (256, 256),
    watermark_length: int = 256,
    num_workers: int = 4,
    augment: bool = False,
    shuffle: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for the watermarking system.

    Args:
        image_dir:        Path to image directory.
        batch_size:       Batch size (16 for RTX 3050, 32 for Kaggle).
        image_size:       Target image dimensions.
        watermark_length: Watermark bit length.
        num_workers:      DataLoader workers.
        augment:          Enable random augmentations.
        shuffle:          Shuffle dataset.
        drop_last:        Drop incomplete final batch (important for BatchNorm).

    Returns:
        Configured DataLoader.
    """
    dataset = WatermarkDataset(
        image_dir=image_dir,
        image_size=image_size,
        watermark_length=watermark_length,
        augment=augment,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,    # Speeds up CPU→GPU transfer
        drop_last=drop_last,
        persistent_workers=num_workers > 0,  # Keep workers alive between epochs
    )
