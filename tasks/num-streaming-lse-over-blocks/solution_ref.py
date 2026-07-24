import numpy as np


def streaming_lse(x: np.ndarray, block_size: int) -> float:
    """Online log-sum-exp: rescale-and-accumulate over sequential blocks."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    m = -np.inf
    s = 0.0
    for start in range(0, n, block_size):
        block = x[start:start + block_size]
        b_max = np.max(block)
        new_m = b_max if b_max > m else m
        s = s * np.exp(m - new_m) + np.sum(np.exp(block - new_m))
        m = new_m
    return float(m + np.log(s))
