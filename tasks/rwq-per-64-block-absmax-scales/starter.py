import numpy as np


def nf4_block_absmax_scales(W: np.ndarray) -> np.ndarray:
    """Flatten W (row-major), split into contiguous 64-element blocks,
    return the per-block max absolute value as a 1-D float64 array."""
    raise NotImplementedError('your code here')
