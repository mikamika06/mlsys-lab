import numpy as np


def lu_partial_pivot(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Factor a square matrix ``A`` as ``P @ A = L @ U``.

    * ``P`` — n x n permutation matrix (float64, entries 0.0/1.0).
    * ``L`` — n x n unit lower-triangular matrix (ones on the diagonal).
    * ``U`` — n x n upper-triangular matrix.

    BUG: this picks the "first nonzero pivot" — it only swaps rows when the
    current diagonal candidate is EXACTLY zero, instead of searching for the
    largest-magnitude candidate. On an ill-scaled matrix (a tiny-but-nonzero
    pivot with much larger entries below it) this produces huge multipliers
    and the L/U factors lose almost all precision, even though A[k, k] != 0
    technically holds.
    """
    A = np.asarray(A, dtype=np.float64).copy()
    n = A.shape[0]
    perm = np.arange(n)
    L = np.zeros((n, n), dtype=np.float64)

    for k in range(n - 1):
        if A[k, k] == 0.0:
            nz = np.nonzero(A[k:, k])[0]
            if len(nz) > 0:
                p = k + int(nz[0])
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
