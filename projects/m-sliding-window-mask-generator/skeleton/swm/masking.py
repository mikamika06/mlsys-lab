import numpy as np


def generate_sliding_window_mask(seq_len: int, window_size: int) -> np.ndarray:
    """
    Generate a boolean causal sliding-window mask of shape (seq_len, seq_len).
    True indicates attention is allowed; False indicates it is blocked.
    """
    raise NotImplementedError
