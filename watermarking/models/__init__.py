"""Models package — exports encoder and decoder."""

from .encoder import WatermarkEncoder
from .decoder import WatermarkDecoder

__all__ = ["WatermarkEncoder", "WatermarkDecoder"]
