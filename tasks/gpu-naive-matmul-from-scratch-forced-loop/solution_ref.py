import numpy as np

def matmul_loops(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute the matrix product C = A @ B using explicit nested loops.
    Parameters
    ----------
    A : np.ndarray
        Left factor of shape (m, k).
    B : np.ndarray
        Right factor of shape (k, n).
    Returns
    -------
    C : np.ndarray
        Resulting matrix of shape (m, n) with dtype float64.
    """
    m, k = A.shape
    _, n = B.shape
    # Allocate output array
    C = np.empty((m, n), dtype=np.float64)
    # Triple nested loop to accumulate products
    for i in range(m):
        for j in range(n):
            acc = 0.0
            for p in range(k):
                acc += A[i, p] * B[p, j]
            C[i, j] = acc
    return C
