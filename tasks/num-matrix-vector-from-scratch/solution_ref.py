import numpy as np


def matvec_from_scratch(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Matrix-vector product via explicit double loop (no dot/matmul)."""
    A = np.asarray(A, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    n, m = A.shape
    y = np.zeros(n, dtype=np.float64)
    for i in range(n):
        total = 0.0
        for j in range(m):
            total += A[i, j] * x[j]
        y[i] = total
    return y
