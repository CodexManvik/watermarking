"""Data package — exports dataset and dataloader factory."""

from .loader import WatermarkDataset, get_dataloader

__all__ = ["WatermarkDataset", "get_dataloader"]
