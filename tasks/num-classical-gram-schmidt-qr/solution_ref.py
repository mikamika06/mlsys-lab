import numpy as np


def classical_gram_schmidt_qr(A):
    A = np.asarray(A, dtype=np.float64)
    m, n = A.shape

    Q = np.zeros((m, n), dtype=np.float64)
    R = np.zeros((n, n), dtype=np.float64)

    for j in range(n):
        u = A[:, j].copy()
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            u = u - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(u)
        Q[:, j] = u / R[j, j]

    return Q, R
