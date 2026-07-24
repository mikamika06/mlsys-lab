import numpy as np


def masked_block_sum(x: np.ndarray, block_size: int) -> np.ndarray:
    x = np.asarray(x)
    n = len(x)
    blocks = (n + block_size - 1) // block_size
    out = np.zeros(blocks, dtype=np.float64)

    for k in range(blocks):
        start = k * block_size
        end = min(start + block_size, n)
        out[k] = np.sum(x[start:end], dtype=np.float64)

    return out
