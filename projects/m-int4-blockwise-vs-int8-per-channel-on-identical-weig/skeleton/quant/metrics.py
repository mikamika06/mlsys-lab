import numpy as np


def compute_mse(w_orig: np.ndarray, w_dequant: np.ndarray) -> float:
    """Computes Mean Squared Error between original weights and dequantized weights."""
    raise NotImplementedError


def compute_bit_size(w_shape: tuple[int, int], mode: str, block_size: int = 32) -> int:
    """Computes total storage bits required for weights and scale metadata."""
    raise NotImplementedError
