import math
import numpy as np


def qr_eigenvalues(A: np.ndarray, max_iter: int = 1000, tol: float = 1e-12) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64).copy()
    n = A.shape[0]

    for _ in range(max_iter):
        Q = np.zeros((n, n), dtype=np.float64)
        R = np.zeros((n, n), dtype=np.float64)

        for j in range(n):
            v = np.zeros(n, dtype=np.float64)
            for i in range(n):
                v[i] = A[i, j]

            for k in range(j):
                q_k = np.zeros(n, dtype=np.float64)
                for i in range(n):
                    q_k[i] = Q[i, k]

                dot_val = 0.0
                for i in range(n):
                    dot_val += q_k[i] * A[i, j]

                R[k, j] = dot_val
                for i in range(n):
                    v[i] -= dot_val * q_k[i]

            norm_v = 0.0
            for i in range(n):
                norm_v += v[i] * v[i]
            norm_v = math.sqrt(norm_v)

            R[j, j] = norm_v
            if norm_v != 0.0:
                for i in range(n):
                    Q[i, j] = v[i] / norm_v

        A_next = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                s = 0.0
                for k in range(n):
                    s += R[i, k] * Q[k, j]
                A_next[i, j] = s
        A = A_next

        off_norm_sq = 0.0
        for i in range(n):
            for j in range(n):
                val = A[i, j]
                if i == j:
                    val = 0.0
                off_norm_sq += val * val

        if math.sqrt(off_norm_sq) < tol:
            break

    res = np.zeros(n, dtype=np.float64)
    for i in range(n):
        res[i] = A[i, i]
    return res
