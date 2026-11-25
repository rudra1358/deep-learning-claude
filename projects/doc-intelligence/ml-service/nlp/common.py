"""Small helpers shared by the NLP models."""
from __future__ import annotations

import torch


def pick_device(prefer: str = "auto") -> str:
    if prefer == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if prefer == "cuda":
        raise RuntimeError("CUDA requested but not available")
    return "cpu"
