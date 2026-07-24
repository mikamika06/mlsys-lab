import numpy as np


def vector_wise_int8_matmul(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """sx = max|X row|/127 (n,); sw = max|W col|/127 (m,). Xq = clip(round(
    X/sx[:,None]),-127,127); Wq = clip(round(W/sw[None,:]),-127,127). acc =
    Xq @ Wq (int accumulate). Return acc * outer(sx, sw), shape (n, m)."""
    raise NotImplementedError('your code here')
