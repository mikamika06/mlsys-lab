import numpy as np


def reconstruct_from_svd(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return U @ np.diag(s) @ Vt
