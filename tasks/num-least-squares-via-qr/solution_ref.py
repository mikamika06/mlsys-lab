import math
import numpy as np


def least_squares_qr(A, b):
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    m, n = A.shape
    R = np.zeros((m, n), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            R[i, j] = float(A[i, j])

    Q = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        Q[i, i] = 1.0

    for k in range(n):
        s_sq = 0.0
        for i in range(k, m):
            s_sq += R[i, k] * R[i, k]
        norm_x = math.sqrt(s_sq)

        if norm_x == 0:
            raise ValueError("rank deficient matrix")

        sign = 1.0 if R[k, k] >= 0 else -1.0
        v_len = m - k
        v = np.zeros(v_len, dtype=np.float64)
        for i in range(v_len):
            v[i] = R[k + i, k]
        v[0] += sign * norm_x

        v_sq = 0.0
        for i in range(v_len):
            v_sq += v[i] * v[i]
        norm_v = math.sqrt(v_sq)

        for i in range(v_len):
            v[i] /= norm_v

        w = np.zeros(n - k, dtype=np.float64)
        for j in range(n - k):
            acc = 0.0
            for i in range(v_len):
                acc += v[i] * R[k + i, k + j]
            w[j] = acc

        for i in range(v_len):
            for j in range(n - k):
                R[k + i, k + j] -= 2.0 * v[i] * w[j]

        u = np.zeros(m, dtype=np.float64)
        for i in range(m):
            acc = 0.0
            for j in range(v_len):
                acc += Q[i, k + j] * v[j]
            u[i] = acc

        for i in range(m):
            for j in range(v_len):
                Q[i, k + j] -= 2.0 * u[i] * v[j]

    y = np.zeros(m, dtype=np.float64)
    for i in range(m):
        acc = 0.0
        for j in range(m):
            acc += Q[j, i] * b[j]
        y[i] = acc

    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        dot_acc = 0.0
        for j in range(i + 1, n):
            dot_acc += R[i, j] * x[j]
        x[i] = (y[i] - dot_acc) / R[i, i]

    return x
