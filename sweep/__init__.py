"""
SWEEP: Single-pass Wavefront Entropy-gated Execution Pipeline
High-Performance Inference Engine for 7B/8B Large Language Models.

Author: Umberto Altieri
License: MIT
"""

__version__ = "2.0.0"
__author__ = "Umberto Altieri"

from .engine import SweepEngine

__all__ = ["SweepEngine", "__version__", "__author__"]
