"""GPU-accelerated CopyKAT-style CNV inference for single-cell RNA-seq."""
from .api import CopyKATResult, copykat
from .backend import resolve_device
__all__ = ["CopyKATResult", "copykat", "resolve_device"]
