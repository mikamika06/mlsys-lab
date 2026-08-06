import math
import numpy as np


def prefix_chunk_hit_rate(prefix_lengths: np.ndarray, chunk_size: int) -> float:
    p = np.asarray(prefix_lengths, dtype=np.float64)
    c = float(chunk_size)
    total = 0.0
    n = len(p)
    for i in range(n):
        val = p[i]
        reused = math.floor(val / c) * c
        total += reused / val
    return float(total / n)
