def lu_partial_pivot(A: list[list[float]]) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    """
    Factor a square matrix ``A`` as ``P @ A = L @ U`` using Gaussian
    elimination with PARTIAL PIVOTING.

    * ``P`` — n x n permutation matrix (entries 0.0/1.0).
    * ``L`` — n x n unit lower-triangular matrix (ones on the diagonal).
    * ``U`` — n x n upper-triangular matrix.

    At elimination step ``k``, the pivot row is chosen as the row in
    ``k .. n-1`` whose entry in column ``k`` has the LARGEST absolute value
    (not merely the first nonzero one). This bounds the size of the
    multipliers used in elimination and keeps the factorization numerically
    stable even when the matrix is ill-scaled.
    """
    n = len(A)
    A_copy = [[float(val) for val in row] for row in A]
    perm = list(range(n))
    L = [[0.0] * n for _ in range(n)]

    for k in range(n - 1):
        max_val = -1.0
        p = k
        for i in range(k, n):
            val = abs(A_copy[i][k])
            if val > max_val:
                max_val = val
                p = i

        if p != k:
            A_copy[k], A_copy[p] = A_copy[p], A_copy[k]
            L[k][:k], L[p][:k] = L[p][:k], L[k][:k]
            perm[k], perm[p] = perm[p], perm[k]

        pivot = A_copy[k][k]
        for i in range(k + 1, n):
            m = A_copy[i][k] / pivot if pivot != 0.0 else 0.0
            L[i][k] = m
            for j in range(k, n):
                A_copy[i][j] -= m * A_copy[k][j]

    for i in range(n):
        L[i][i] = 1.0
    U = A_copy

    P = [[0.0] * n for _ in range(n)]
    for i in range(n):
        P[i][perm[i]] = 1.0

    return P, L, U
