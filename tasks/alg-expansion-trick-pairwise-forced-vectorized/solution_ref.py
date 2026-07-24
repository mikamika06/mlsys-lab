import numpy as np


def pairwise_sq_dists(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    g = np.sum(A * A, axis=1)
    return g[:, None] + g[None, :] - 2.0 * (A @ A.T)
