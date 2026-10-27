"""Device resolution utility supporting CPU, CUDA, and Apple Silicon MPS."""

from typing import Literal

DeviceType = Literal["auto", "cpu", "cuda", "mps"]


def resolve_device(device: str = "auto") -> str:
    """Resolve device preference to an available runtime device string.

    Args:
        device: Requested device ('auto', 'cpu', 'cuda', 'mps').

    Returns:
        Device string suitable for the backend ('cpu', 'cuda', 'cuda:0', 'mps').
    """
    device_lower = str(device).lower().strip()

    if device_lower in ("cpu", "cuda", "mps"):
        # Explicit request
        if device_lower == "cuda":
            try:
                import torch

                if not torch.cuda.is_available():
                    return "cpu"
                return "cuda"
            except ImportError:
                return "cpu"
        elif device_lower == "mps":
            try:
                import torch

                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return "mps"
                return "cpu"
            except ImportError:
                return "cpu"
        return device_lower

    # Automatic selection
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass

    return "cpu"
