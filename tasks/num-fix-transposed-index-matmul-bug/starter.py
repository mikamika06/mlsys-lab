import numpy as np


def matmul_naive(A, B):
    """Compute A @ B for A of shape (m,k) and B of shape (k,n) using explicit
    Python loops. Returns a float64 NumPy array of shape (m,n)."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    m, k = A.shape
    k2, n = B.shape
    C = np.zeros((m, n), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            s = 0.0
            for p in range(k):
                s += A[i, p] * B[j, p]          # ← BUG: should be B[p, j]
            C[i, j] = s
    return C
