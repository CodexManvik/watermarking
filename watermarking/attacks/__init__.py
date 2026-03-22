"""Attacks package — exports the multi-branch attack simulator."""

from .simulator import AttackSimulator
from .jpeg_diff import DifferentiableJPEG

__all__ = ["AttackSimulator", "DifferentiableJPEG"]
