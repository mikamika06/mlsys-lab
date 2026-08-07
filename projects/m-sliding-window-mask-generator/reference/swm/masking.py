import numpy as np


def generate_sliding_window_mask(seq_len: int, window_size: int) -> np.ndarray:
    i = np.arange(seq_len)[:, None]
    j = np.arange(seq_len)[None, :]
    dist = i - j
    return (dist >= 0) & (dist < window_size)
