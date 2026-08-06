import math
import numpy as np


def streaming_lse(x: np.ndarray, block_size: int) -> float:
    """Online log-sum-exp: rescale-and-accumulate over sequential blocks."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    m = -float("inf")
    s = 0.0
    for start in range(0, n, block_size):
        end = start + block_size
        if end > n:
            end = n
        b_max = -float("inf")
        for i in range(start, end):
            val = float(x[i])
            if val > b_max:
                b_max = val
        new_m = b_max if b_max > m else m
        s = s * math.exp(m - new_m)
        for i in range(start, end):
            s += math.exp(float(x[i]) - new_m)
        m = new_m
    return float(m + math.log(s))
