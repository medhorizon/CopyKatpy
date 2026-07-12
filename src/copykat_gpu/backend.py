"""PyTorch backend helpers."""
from __future__ import annotations
import torch

def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Return an available compute device, preferring CUDA when possible."""
    if device is None or str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = torch.device(device)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA device.")
    return selected
