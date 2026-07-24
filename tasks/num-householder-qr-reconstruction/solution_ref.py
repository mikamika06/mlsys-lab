import numpy as np


def householder_qr_reconstruct(A):
    A = np.asarray(A, dtype=np.float64)
    m, n = A.shape
    R = A.copy()
    Q = np.eye(m, dtype=np.float64)

    for k in range(min(m, n)):
        x = R[k:, k].copy()
        norm = np.linalg.norm(x)
        if norm == 0:
            continue

        sign = -1.0 if x[0] < 0 else 1.0
        v = x.copy()
        v[0] += sign * norm

        vnorm = np.linalg.norm(v)
        if vnorm == 0:
            continue
        v /= vnorm

        R[k:, k:] -= 2.0 * np.outer(v, v @ R[k:, k:])
        Q[:, k:] -= 2.0 * np.outer(Q[:, k:] @ v, v)

    return Q, R
