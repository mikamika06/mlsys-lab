import numpy as np


def compress_svd(A: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return compact rank-k SVD factors."""
    raise NotImplementedError("your code here")


def reconstruct_svd(U: np.ndarray, S: np.ndarray, Vt: np.ndarray) -> np.ndarray:
    """Reconstruct a matrix from compact SVD factors."""
    raise NotImplementedError("your code here")
