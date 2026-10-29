"""
SynapticFold Native Kernels Module.
"""
from .native_router import is_native_avx2_available, compute_column_potentials_avx2

__all__ = ["is_native_avx2_available", "compute_column_potentials_avx2"]
