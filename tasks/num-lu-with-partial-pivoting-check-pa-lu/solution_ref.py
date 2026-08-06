import math


def lu_partial_pivot(A: list[list[float]]) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    """LU decomposition with partial pivoting."""
    n = len(A)

    M = [[A[i][j] for j in range(n)] for i in range(n)]
    P = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    L = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for k in range(n - 1):
        max_val = -1.0
        max_idx = 0
        for idx, i in enumerate(range(k, n)):
            val = math.fabs(M[i][k])
            if val > max_val:
                max_val = val
                max_idx = idx
        pivot = k + max_idx

        if pivot != k:
            for col in range(n):
                temp = M[k][col]
                M[k][col] = M[pivot][col]
                M[pivot][col] = temp
            for col in range(n):
                temp = P[k][col]
                P[k][col] = P[pivot][col]
                P[pivot][col] = temp
            if k > 0:
                for col in range(k):
                    temp = L[k][col]
                    L[k][col] = L[pivot][col]
                    L[pivot][col] = temp

        for i in range(k + 1, n):
            L[i][k] = M[i][k] / M[k][k]
            for col in range(k, n):
                M[i][col] -= L[i][k] * M[k][col]

    return P, L, M
