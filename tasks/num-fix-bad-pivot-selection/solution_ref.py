import numpy as np


def lu_partial_pivot(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Factor a square matrix ``A`` as ``P @ A = L @ U`` using Gaussian
    elimination with PARTIAL PIVOTING.

    * ``P`` — n x n permutation matrix (float64, entries 0.0/1.0).
    * ``L`` — n x n unit lower-triangular matrix (ones on the diagonal).
    * ``U`` — n x n upper-triangular matrix.

    At elimination step ``k``, the pivot row is chosen as the row in
    ``k .. n-1`` whose entry in column ``k`` has the LARGEST absolute value
    (not merely the first nonzero one). This bounds the size of the
    multipliers used in elimination and keeps the factorization numerically
    stable even when the matrix is ill-scaled.
    """
    A = np.asarray(A, dtype=np.float64).copy()
    n = A.shape[0]
    perm = list(range(n))
    L = np.zeros((n, n), dtype=np.float64)

    for k in range(n - 1):
        max_val = abs(A[k, k])
        p = k
        for i in range(k + 1, n):
            val = abs(A[i, k])
            if val > max_val:
                max_val = val
                p = i

        if p != k:
            for col in range(n):
                temp = A[k, col]
                A[k, col] = A[p, col]
                A[p, col] = temp
            for col in range(k):
                temp = L[k, col]
                L[k, col] = L[p, col]
                L[p, col] = temp
            temp_perm = perm[k]
            perm[k] = perm[p]
            perm[p] = temp_perm

        pivot = A[k, k]
        for i in range(k + 1, n):
            m = A[i, k] / pivot if pivot != 0.0 else 0.0
            L[i, k] = m
            for j in range(k, n):
                A[i, j] -= m * A[k, j]

    for i in range(n):
        L[i, i] = 1.0

    U = A
    P = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        P[i, perm[i]] = 1.0

    return P, L, U
