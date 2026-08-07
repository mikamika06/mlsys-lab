import numpy as np


def compute_scales(tensor: np.ndarray, mode: str, block_size: int = 32) -> np.ndarray:
    """Compute quantization scales."""
    raise NotImplementedError
