import numpy as np


def low_rank_factors(W: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return thin factors (A, B) with A = U_k @ Sigma_k, B = V_k^T."""
    W = np.asarray(W, dtype=np.float64)
    u, s, vt = np.linalg.svd(W, full_matrices=False)
    A = u[:, :k] * s[:k]
    B = vt[:k, :]
    return np.ascontiguousarray(A, dtype=np.float64), np.ascontiguousarray(B, dtype=np.float64)


def low_rank_reconstruct(W: np.ndarray, k: int) -> np.ndarray:
    """Return the optimal rank-k approximation A @ B of W."""
    A, B = low_rank_factors(W, k)
    return A @ B
