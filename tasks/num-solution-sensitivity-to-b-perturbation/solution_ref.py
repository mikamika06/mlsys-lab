import math
import numpy as np


def solution_sensitivity(A, b, delta):
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)

    n = len(b)
    b_p = [float(b[i]) + float(delta[i]) for i in range(n)]

    def solve_system(A_mat, rhs):
        M = [[float(A_mat[i, j]) for j in range(n)] for i in range(n)]
        v = [float(rhs[i]) for i in range(n)]

        for i in range(n):
            max_row = i
            max_val = M[i][i] if M[i][i] >= 0.0 else -M[i][i]
            for k in range(i + 1, n):
                val = M[k][i] if M[k][i] >= 0.0 else -M[k][i]
                if val > max_val:
                    max_val = val
                    max_row = k

            if max_row != i:
                M[i], M[max_row] = M[max_row], M[i]
                v[i], v[max_row] = v[max_row], v[i]

            for k in range(i + 1, n):
                factor = M[k][i] / M[i][i]
                for j in range(i, n):
                    M[k][j] -= factor * M[i][j]
                v[k] -= factor * v[i]

        sol = [0.0] * n
        for i in range(n - 1, -1, -1):
            s = v[i]
            for j in range(i + 1, n):
                s -= M[i][j] * sol[j]
            sol[i] = s / M[i][i]

        return sol

    x = solve_system(A, b)
    xp = solve_system(A, b_p)

    diff_sq_sum = 0.0
    x_sq_sum = 0.0
    for i in range(n):
        diff = xp[i] - x[i]
        diff_sq_sum += diff * diff
        x_sq_sum += x[i] * x[i]

    norm_diff = math.sqrt(diff_sq_sum)
    norm_x = math.sqrt(x_sq_sum)

    return float(norm_diff / (norm_x + 1e-12))
