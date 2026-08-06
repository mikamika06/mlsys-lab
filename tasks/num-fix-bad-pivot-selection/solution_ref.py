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
    perm = np.arange(n)
    L = np.zeros((n, n), dtype=np.float64)

    for k in range(n - 1):
        # partial pivoting: largest-magnitude entry in column k, rows k..n-1
        p = k + int(np.argmax(np.abs(A[k:, k])))
        if p != k:
            A[[k, p], :] = A[[p, k], :]
            L[[k, p], :k] = L[[p, k], :k]
            perm[[k, p]] = perm[[p, k]]

        pivot = A[k, k]
        for i in range(k + 1, n):
            m = A[i, k] / pivot if pivot != 0.0 else 0.0
            L[i, k] = m
            A[i, k:] -= m * A[k, k:]

    np.fill_diagonal(L, 1.0)
    U = A
    P = np.eye(n, dtype=np.float64)[perm]
    return P, L, U
