from __future__ import annotations

import math
import numpy as np


def inverse_iteration(A: np.ndarray, num_iters: int = 100, x0: np.ndarray | None = None):
    """Inverse iteration: converges to the eigenpair of smallest |eigenvalue|."""
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]

    if x0 is None:
        x = np.zeros(n, dtype=np.float64)
        val = 1.0 / math.sqrt(n)
        for i in range(n):
            x[i] = val
    else:
        x = np.zeros(n, dtype=np.float64)
        x0_arr = np.asarray(x0, dtype=np.float64).ravel()
        for i in range(n):
            x[i] = x0_arr[i]

    sq_sum = 0.0
    for i in range(n):
        sq_sum += x[i] * x[i]
    norm_x = math.sqrt(sq_sum) + 1e-15
    for i in range(n):
        x[i] /= norm_x

    LU = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            LU[i, j] = float(A[i, j])

    piv = [i for i in range(n)]

    for i in range(n):
        max_val = 0.0
        max_row = i
        for r in range(i, n):
            val = LU[r, i]
            if val < 0.0:
                val = -val
            if val > max_val:
                max_val = val
                max_row = r

        if max_row != i:
            piv[i], piv[max_row] = piv[max_row], piv[i]
            for c in range(n):
                LU[i, c], LU[max_row, c] = LU[max_row, c], LU[i, c]

        pivot_val = LU[i, i]
        if pivot_val != 0.0:
            for r in range(i + 1, n):
                factor = LU[r, i] / pivot_val
                LU[r, i] = factor
                for c in range(i + 1, n):
                    LU[r, c] -= factor * LU[i, c]

    for _ in range(num_iters):
        b = np.zeros(n, dtype=np.float64)
        for i in range(n):
            b[i] = x[piv[i]]

        y_prime = np.zeros(n, dtype=np.float64)
        for i in range(n):
            s = b[i]
            for j in range(i):
                s -= LU[i, j] * y_prime[j]
            y_prime[i] = s

        y = np.zeros(n, dtype=np.float64)
        for i in range(n - 1, -1, -1):
            s = y_prime[i]
            for j in range(i + 1, n):
                s -= LU[i, j] * y[j]
            if LU[i, i] != 0.0:
                y[i] = s / LU[i, i]
            else:
                y[i] = s

        sq_sum = 0.0
        for i in range(n):
            sq_sum += y[i] * y[i]
        norm_y = math.sqrt(sq_sum) + 1e-15
        for i in range(n):
            x[i] = y[i] / norm_y

    Ax = np.zeros(n, dtype=np.float64)
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += float(A[i, j]) * x[j]
        Ax[i] = s

    eigval = 0.0
    for i in range(n):
        eigval += x[i] * Ax[i]

    return float(eigval), x
