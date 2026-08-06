import math
import numpy as np


def wilkinson_eigvals(A: np.ndarray, max_iter: int = 200) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64).copy()
    m = A.shape[0]
    eps = 1e-12

    for _ in range(max_iter):
        if m <= 1:
            break

        if abs(A[m - 1, m - 2]) <= eps:
            A[m - 1, m - 2] = 0.0
            A[m - 2, m - 1] = 0.0
            m -= 1
            continue

        a = A[m - 2, m - 2]
        b = A[m - 2, m - 1]
        d = A[m - 1, m - 1]

        delta = (a - d) / 2.0
        sign = 1.0 if delta >= 0 else -1.0
        mu = d - (b * b) / (delta + sign * math.sqrt(delta * delta + b * b))

        B = [[0.0] * m for _ in range(m)]
        for i in range(m):
            for j in range(m):
                val = A[i, j]
                if i == j:
                    val -= mu
                B[i][j] = val

        R = [[B[i][j] for j in range(m)] for i in range(m)]
        Q = [[1.0 if i == j else 0.0 for j in range(m)] for i in range(m)]

        for k in range(m - 1):
            s_sq = 0.0
            for i in range(k, m):
                s_sq += R[i][k] * R[i][k]
            norm_x = math.sqrt(s_sq)
            if norm_x == 0.0:
                continue

            rk = R[k][k]
            alpha = -norm_x if rk >= 0 else norm_x

            v = [0.0] * m
            v[k] = rk - alpha
            for i in range(k + 1, m):
                v[i] = R[i][k]

            norm_v_sq = 0.0
            for i in range(k, m):
                norm_v_sq += v[i] * v[i]

            if norm_v_sq == 0.0:
                continue

            vT_R = [0.0] * m
            for j in range(m):
                s = 0.0
                for i in range(k, m):
                    s += v[i] * R[i][j]
                vT_R[j] = s

            factor = 2.0 / norm_v_sq
            for i in range(k, m):
                vi = v[i]
                if vi == 0.0:
                    continue
                for j in range(m):
                    R[i][j] -= factor * vi * vT_R[j]

            Q_v = [0.0] * m
            for i in range(m):
                s = 0.0
                for j in range(k, m):
                    s += Q[i][j] * v[j]
                Q_v[i] = s

            for i in range(m):
                qvi = Q_v[i]
                if qvi == 0.0:
                    continue
                for j in range(k, m):
                    Q[i][j] -= factor * qvi * v[j]

        RQ = [[0.0] * m for _ in range(m)]
        for i in range(m):
            for j in range(m):
                s = 0.0
                for k in range(m):
                    s += R[i][k] * Q[k][j]
                RQ[i][j] = s

        for i in range(m):
            for j in range(m):
                val = RQ[i][j]
                if i == j:
                    val += mu
                A[i, j] = val

    diag_vals = []
    for i in range(A.shape[0]):
        diag_vals.append(A[i, i])

    n_diag = len(diag_vals)
    for i in range(n_diag):
        for j in range(0, n_diag - i - 1):
            if diag_vals[j] > diag_vals[j + 1]:
                diag_vals[j], diag_vals[j + 1] = diag_vals[j + 1], diag_vals[j]

    return np.array(diag_vals, dtype=np.float64)
