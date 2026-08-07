import numpy as np


def windowed_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Compute scaled dot-product attention using the provided boolean mask.
    Blocked positions must evaluate to -1e9 before softmax.
    """
    raise NotImplementedError
