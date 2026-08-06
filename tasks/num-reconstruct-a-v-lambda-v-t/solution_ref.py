import math
import numpy as np


def reconstruct_from_eigh(A: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    mat = [[float(A[i, j]) for j in range(n)] for i in range(n)]
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for _ in range(100):
        max_val = 0.0
        p, q = 0, 1
        for i in range(n):
            for j in range(i + 1, n):
                val = math.fabs(mat[i][j])
                if val > max_val:
                    max_val = val
                    p, q = i, j

        if max_val < 1e-15:
            break

        app = mat[p][p]
        aqq = mat[q][q]
        apq = mat[p][q]

        if math.fabs(app - aqq) < 1e-15:
            theta = math.pi / 4.0 if apq > 0 else -math.pi / 4.0
        else:
            tau = (aqq - app) / (2.0 * apq)
            t = 1.0 / (math.fabs(tau) + math.sqrt(1.0 + tau * tau))
            if tau < 0:
                t = -t
            theta = math.atan(t)

        c = math.cos(theta)
        s = math.sin(theta)

        J = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        J[p][p] = c
        J[q][q] = c
        J[p][q] = s
        J[q][p] = -s

        JT = [[J[row][col] for row in range(n)] for col in range(n)]
        JT_mat = [[sum(JT[i][k] * mat[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
        mat = [[sum(JT_mat[i][k] * J[k][j] for k in range(n)) for j in range(n)] for i in range(n)]

        V_new = [[sum(V[i][k] * J[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
        V = V_new

    eigenpairs = []
    for i in range(n):
        w_i = mat[i][i]
        v_i = [V[row][i] for row in range(n)]
        eigenpairs.append((w_i, v_i))

    eigenpairs = sorted(eigenpairs, key=lambda x: x[0])

    w = [ep[0] for ep in eigenpairs]
    V_sorted = [[0.0 for _ in range(n)] for _ in range(n)]
    for j in range(n):
        for i in range(n):
            V_sorted[i][j] = eigenpairs[j][1][i]

    result = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s_val = 0.0
            for k in range(n):
                s_val += V_sorted[i][k] * w[k] * V_sorted[j][k]
            result[i][j] = s_val

    return np.array(result, dtype=A.dtype)
