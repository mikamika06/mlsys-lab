import math
import numpy as np


def shifted_inverse_iteration(A, mu, x0, iters):
    n = A.shape[0]
    x = [x0[i] for i in range(n)]

    norm_sq = 0.0
    for i in range(n):
        norm_sq += x[i] * x[i]
    norm = math.sqrt(norm_sq)
    for i in range(n):
        x[i] /= norm

    shifted = [[A[i, j] - (mu if i == j else 0.0) for j in range(n)] for i in range(n)]

    for _ in range(iters):
        M = [row[:] for row in shifted]
        b = list(x)

        for i in range(n):
            max_val = abs(M[i][i])
            max_row = i
            for k in range(i + 1, n):
                val = abs(M[k][i])
                if val > max_val:
                    max_val = val
                    max_row = k

            if max_row != i:
                M[i], M[max_row] = M[max_row], M[i]
                b[i], b[max_row] = b[max_row], b[i]

            pivot = M[i][i]
            for k in range(i + 1, n):
                factor = M[k][i] / pivot
                M[k][i] = 0.0
                for j in range(i + 1, n):
                    M[k][j] -= factor * M[i][j]
                b[k] -= factor * b[i]

        y = [0.0] * n
        for i in range(n - 1, -1, -1):
            s = b[i]
            for j in range(i + 1, n):
                s -= M[i][j] * y[j]
            y[i] = s / M[i][i]

        norm_sq = 0.0
        for i in range(n):
            norm_sq += y[i] * y[i]
        norm = math.sqrt(norm_sq)
        for i in range(n):
            x[i] = y[i] / norm

    xA = [0.0] * n
    for j in range(n):
        s = 0.0
        for i in range(n):
            s += x[i] * A[i, j]
        xA[j] = s

    num = 0.0
    for j in range(n):
        num += xA[j] * x[j]

    den = 0.0
    for i in range(n):
        den += x[i] * x[i]

    return float(num / den)
