import numpy as np


def solve_lu(A, b):
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    n = A.shape[0]
    U = A.copy()
    L = np.eye(n, dtype=np.float64)
    piv = np.arange(n)

    for k in range(n - 1):
        p = k + np.argmax(np.abs(U[k:, k]))
        if p != k:
            U[[k, p]] = U[[p, k]]
            L[[k, p], :k] = L[[p, k], :k]
            piv[[k, p]] = piv[[p, k]]

        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]

    y = b[piv].copy()
    for i in range(n):
        y[i] -= np.dot(L[i, :i], y[:i])

    x = y.copy()
    for i in range(n - 1, -1, -1):
        x[i] -= np.dot(U[i, i + 1:], x[i + 1:])
        x[i] /= U[i, i]

    return x
