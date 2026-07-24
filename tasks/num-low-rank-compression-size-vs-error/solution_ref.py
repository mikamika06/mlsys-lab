import numpy as np


def compress_svd(A: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    A = np.asarray(A, dtype=np.float64)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    return U[:, :k], S[:k], Vt[:k, :]


def reconstruct_svd(U: np.ndarray, S: np.ndarray, Vt: np.ndarray) -> np.ndarray:
    return (U * S) @ Vt
