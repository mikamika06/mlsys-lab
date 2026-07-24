import numpy as np


def matmul_ikj(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Matrix multiply via explicit i, k, j nested loops (row-streaming)."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    n, k_dim = A.shape
    k_dim2, m = B.shape
    assert k_dim == k_dim2
    C = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        for k in range(k_dim):
            a_ik = A[i, k]
            for j in range(m):
                C[i, j] += a_ik * B[k, j]
    return C
