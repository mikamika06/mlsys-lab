import math
import numpy as np


def rayleigh_quotient_iteration(A: np.ndarray, v0: np.ndarray, n_iter: int) -> float:
    """Rayleigh quotient iteration: adaptive-shift inverse iteration, cubic convergence."""
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    v = np.asarray(v0, dtype=np.float64)
    
    sum_sq = 0.0
    for i in range(n):
        sum_sq += v[i] * v[i]
    norm_v = math.sqrt(sum_sq)
    for i in range(n):
        v[i] /= norm_v

    mu = 0.0
    for i in range(n):
        row_sum = 0.0
        for j in range(n):
            row_sum += A[i, j] * v[j]
        mu += v[i] * row_sum
    mu = float(mu)

    for _ in range(n_iter):
        try:
            M = np.empty((n, n), dtype=np.float64)
            for i in range(n):
                for j in range(n):
                    M[i, j] = A[i, j]
                M[i, i] -= mu
            
            b = np.empty(n, dtype=np.float64)
            for i in range(n):
                b[i] = v[i]

            for i in range(n):
                pivot = i
                max_val = abs(M[i, i])
                for k in range(i + 1, n):
                    val = abs(M[k, i])
                    if val > max_val:
                        max_val = val
                        pivot = k
                
                if pivot != i:
                    for j in range(i, n):
                        tmp = M[i, j]
                        M[i, j] = M[pivot, j]
                        M[pivot, j] = tmp
                    tmp = b[i]
                    b[i] = b[pivot]
                    b[pivot] = tmp

                for k in range(i + 1, n):
                    factor = M[k, i] / M[i, i]
                    M[k, i] = 0.0
                    for j in range(i + 1, n):
                        M[k, j] -= factor * M[i, j]
                    b[k] -= factor * b[i]

            w = np.empty(n, dtype=np.float64)
            for i in range(n - 1, -1, -1):
                s = b[i]
                for j in range(i + 1, n):
                    s -= M[i, j] * w[j]
                w[i] = s / M[i, i]

        except Exception:
            w = np.empty(n, dtype=np.float64)
            for i in range(n):
                w[i] = v[i]

        sum_sq_w = 0.0
        for i in range(n):
            sum_sq_w += w[i] * w[i]
        nrm = math.sqrt(sum_sq_w)

        if nrm == 0.0 or not math.isfinite(nrm):
            break

        for i in range(n):
            v[i] = w[i] / nrm

        mu = 0.0
        for i in range(n):
            row_sum = 0.0
            for j in range(n):
                row_sum += A[i, j] * v[j]
            mu += v[i] * row_sum
        mu = float(mu)

    return mu
