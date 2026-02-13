from __future__ import annotations

from typing import Literal


def resolve_device(device: str) -> str:
    """Resolve 'auto' to the best available device: cuda > mps > cpu."""
    d = (device or "auto").strip().lower()
    if d != "auto":
        return device  # allow "cpu", "mps", "cuda", "0", etc.

    # Try torch if available (best signal for cuda/mps)
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "cuda"
        # MPS available on Apple Silicon if torch built with MPS
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore
            return "mps"
    except Exception:
        pass

    return "cpu"