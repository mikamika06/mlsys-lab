import numpy as np


def lu_partial_pivot(A):
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]

    M = A.copy()
    P = np.eye(n, dtype=np.float64)
    L = np.eye(n, dtype=np.float64)

    for k in range(n - 1):
        pivot = k + int(np.argmax(np.abs(M[k:, k])))

        if pivot != k:
            M[[k, pivot], :] = M[[pivot, k], :]
            P[[k, pivot], :] = P[[pivot, k], :]
            if k > 0:
                L[[k, pivot], :k] = L[[pivot, k], :k]

        for i in range(k + 1, n):
            L[i, k] = M[i, k] / M[k, k]
            M[i, k:] -= L[i, k] * M[k, k:]

    return P, L, M
