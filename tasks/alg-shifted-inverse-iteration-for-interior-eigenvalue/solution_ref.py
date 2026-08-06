import math
import numpy as np


def shifted_inverse_iteration(A, sigma, iterations):
    """Computes the closest eigenvalue and eigenvector using shifted inverse iteration."""
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]

    x = np.empty(n, dtype=np.float64)
    norm_x = math.sqrt(float(n))
    for i in range(n):
        x[i] = 1.0 / norm_x

    M = []
    for i in range(n):
        row = []
        for j in range(n):
            val = float(A[i, j])
            if i == j:
                val -= float(sigma)
            row.append(val)
        M.append(row)

    ipiv = [0] * n
    for i in range(n):
        max_val = -1.0
        pivot_row = i
        for r in range(i, n):
            val = abs(M[r][i])
            if val > max_val:
                max_val = val
                pivot_row = r
        ipiv[i] = pivot_row
        if pivot_row != i:
            M[i], M[pivot_row] = M[pivot_row], M[i]
        pivot = M[i][i]
        for r in range(i + 1, n):
            M[r][i] /= pivot
            factor = M[r][i]
            for c in range(i + 1, n):
                M[r][c] -= factor * M[i][c]

    for _ in range(iterations):
        b = [float(x[k]) for k in range(n)]

        for i in range(n):
            p = ipiv[i]
            if p != i:
                b[i], b[p] = b[p], b[i]
            for j in range(i):
                b[i] -= M[i][j] * b[j]

        for i in range(n - 1, -1, -1):
            for j in range(i + 1, n):
                b[i] -= M[i][j] * b[j]
            b[i] /= M[i][i]

        sq_sum = 0.0
        for k in range(n):
            sq_sum += b[k] * b[k]
        norm_y = math.sqrt(sq_sum)

        for k in range(n):
            x[k] = b[k] / norm_y

    temp = [0.0] * n
    for j in range(n):
        s = 0.0
        for i in range(n):
            s += float(x[i]) * float(A[i, j])
        temp[j] = s

    num = 0.0
    for j in range(n):
        num += temp[j] * float(x[j])

    den = 0.0
    for k in range(n):
        den += float(x[k]) * float(x[k])

    value = float(num / den)
    return value, x
