import numpy as np


def nf4_block_absmax_scales(W: np.ndarray) -> np.ndarray:
    w = np.asarray(W, dtype=np.float64).reshape(-1)
    blocks = w.reshape(-1, 64)
    return np.max(np.abs(blocks), axis=1)
