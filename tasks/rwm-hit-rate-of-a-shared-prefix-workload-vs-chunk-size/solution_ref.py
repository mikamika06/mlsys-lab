import numpy as np


def prefix_chunk_hit_rate(prefix_lengths: np.ndarray, chunk_size: int) -> float:
    p = np.asarray(prefix_lengths, dtype=np.float64)
    c = float(chunk_size)
    reused = np.floor(p / c) * c
    return float(np.mean(reused / p))
