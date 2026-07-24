import numpy as np


def least_squares_qr(A, b):
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    m, n = A.shape
    R = A.copy()
    Q = np.eye(m, dtype=np.float64)

    for k in range(n):
        x = R[k:, k]
        norm_x = np.linalg.norm(x)
        if norm_x == 0:
            raise ValueError("rank deficient matrix")

        sign = 1.0 if x[0] >= 0 else -1.0
        v = x.copy()
        v[0] += sign * norm_x
        v /= np.linalg.norm(v)

        R[k:, k:] -= 2.0 * np.outer(v, v @ R[k:, k:])
        Q[:, k:] -= 2.0 * np.outer(Q[:, k:] @ v, v)

    y = Q.T @ b
    x = np.zeros(n, dtype=np.float64)

    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(R[i, i + 1:n], x[i + 1:n])) / R[i, i]

    return x
