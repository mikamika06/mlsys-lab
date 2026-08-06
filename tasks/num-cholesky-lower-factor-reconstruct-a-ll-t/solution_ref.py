import math
import numpy as np

def cholesky_lower(A: np.ndarray) -> np.ndarray:
    """Cholesky-Banachiewicz: A = L L^T with L lower triangular, L_ii > 0."""
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    L = np.zeros((n, n), dtype=np.float64)

    for j in range(n):
        dot_jj = 0.0
        for k in range(j):
            dot_jj += L[j, k] * L[j, k]
        s = A[j, j] - dot_jj
        
        if s <= 0.0:
            raise ValueError("matrix is not positive definite")
            
        L[j, j] = math.sqrt(s)
        
        if j + 1 < n:
            for i in range(j + 1, n):
                dot_ij = 0.0
                for k in range(j):
                    dot_ij += L[i, k] * L[j, k]
                L[i, j] = (A[i, j] - dot_ij) / L[j, j]

    return L
