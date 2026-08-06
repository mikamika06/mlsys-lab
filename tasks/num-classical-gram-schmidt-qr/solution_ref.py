import math
import numpy as np


def classical_gram_schmidt_qr(A):
    A = np.asarray(A, dtype=np.float64)
    m, n = A.shape

    Q = np.zeros((m, n), dtype=np.float64)
    R = np.zeros((n, n), dtype=np.float64)

    for j in range(n):
        u = np.zeros(m, dtype=np.float64)
        for k in range(m):
            u[k] = A[k, j]
        for i in range(j):
            dot_val = 0.0
            for k in range(m):
                dot_val += Q[k, i] * A[k, j]
            R[i, j] = dot_val
            rij = R[i, j]
            for k in range(m):
                u[k] = u[k] - rij * Q[k, i]
        sum_sq = 0.0
        for k in range(m):
            sum_sq += u[k] * u[k]
        R[j, j] = math.sqrt(sum_sq)
        norm_val = R[j, j]
        for k in range(m):
            Q[k, j] = u[k] / norm_val

    return Q, R
