import numpy as np


def nf4_dequantize(idx: np.ndarray, absmax: np.ndarray, block_size: int = 64) -> np.ndarray:
    """Reconstruct a tensor from stored NF4 4-bit codes + per-block absmax.

    Args:
        idx: 1-D uint8 array, length n (multiple of block_size), entries in [0, 15].
        absmax: 1-D array, length n // block_size, per-block scale.
        block_size: elements per block (default 64).

    Returns:
        np.ndarray of shape (n,): dequant[i] = NF4_LEVELS[idx[i]] * absmax[i // block_size].
    """
    raise NotImplementedError('your code here')
