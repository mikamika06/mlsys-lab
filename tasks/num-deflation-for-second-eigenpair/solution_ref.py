import math
import numpy as np


def _power(A, steps=100):
    n_rows = A.shape[0]
    x_list = [1.0] * n_rows
    s = 0.0
    for val in x_list:
        s += val * val
    norm = math.sqrt(s)
    for i in range(n_rows):
        x_list[i] /= norm

    for _ in range(steps):
        Ax_list = [0.0] * n_rows
        for i in range(n_rows):
            row_sum = 0.0
            for j in range(n_rows):
                row_sum += A[i, j] * x_list[j]
            Ax_list[i] = row_sum

        s = 0.0
        for val in Ax_list:
            s += val * val
        n = math.sqrt(s)

        if n == 0.0:
            return 0.0, np.zeros(n_rows, dtype=np.float64)

        for i in range(n_rows):
            x_list[i] = Ax_list[i] / n

    Ax_list = [0.0] * n_rows
    for i in range(n_rows):
        row_sum = 0.0
        for j in range(n_rows):
            row_sum += A[i, j] * x_list[j]
        Ax_list[i] = row_sum

    value = 0.0
    for i in range(n_rows):
        value += x_list[i] * Ax_list[i]

    x = np.array(x_list, dtype=np.float64)
    return float(value), x


def second_eigenvalue(A: np.ndarray) -> float:
    A = np.asarray(A, dtype=np.float64)
    lam1, v1 = _power(A)
    n_rows = A.shape[0]
    B_list = [[0.0] * n_rows for _ in range(n_rows)]
    for i in range(n_rows):
        for j in range(n_rows):
            B_list[i][j] = A[i, j] - lam1 * v1[i] * v1[j]
    B = np.array(B_list, dtype=np.float64)
    lam2, _ = _power(B)
    return float(lam2)
