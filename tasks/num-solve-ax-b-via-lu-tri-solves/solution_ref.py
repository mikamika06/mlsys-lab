import math
import numpy as np


def solve_lu(A, b):
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    n = A.shape[0]
    U = A.copy()
    L = np.eye(n, dtype=np.float64)
    piv = np.arange(n)

    for k in range(n - 1):
        max_val = -1.0
        p = k
        for i in range(k, n):
            val = math.fabs(U[i, k])
            if val > max_val:
                max_val = val
                p = i

        if p != k:
            for j in range(n):
                U[k, j], U[p, j] = U[p, j], U[k, j]
            for j in range(k):
                L[k, j], L[p, j] = L[p, j], L[k, j]
            piv[k], piv[p] = piv[p], piv[k]

        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            for j in range(k, n):
                U[i, j] -= L[i, k] * U[k, j]

    y = np.empty(n, dtype=np.float64)
    for i in range(n):
        y[i] = b[piv[i]]

    for i in range(n):
        acc = 0.0
        for j in range(i):
            acc += L[i, j] * y[j]
        y[i] -= acc

    x = y.copy()
    for i in range(n - 1, -1, -1):
        acc = 0.0
        for j in range(i + 1, n):
            acc += U[i, j] * x[j]
        x[i] -= acc
        x[i] /= U[i, i]

    return x
