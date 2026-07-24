import numpy as np


def low_rank_factors(W: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return thin factors (A, B) of shapes (m, k) and (k, n) from the truncated SVD of W."""
    raise NotImplementedError('your code here')


def low_rank_reconstruct(W: np.ndarray, k: int) -> np.ndarray:
    """Return the rank-k truncated-SVD reconstruction of W."""
    raise NotImplementedError('your code here')
