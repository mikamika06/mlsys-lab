import math
import numpy as np


def compress_svd(A: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    A = np.asarray(A, dtype=np.float64)
    m, n = A.shape
    U_list = []
    S_list = []
    Vt_list = []

    A_curr = np.zeros((m, n), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            A_curr[i, j] = A[i, j]

    for idx in range(k):
        v = np.zeros((n,), dtype=np.float64)
        for j in range(n):
            v[j] = 1.0 / math.sqrt(n)

        for _ in range(200):
            u = np.zeros((m,), dtype=np.float64)
            for i in range(m):
                acc = 0.0
                for j in range(n):
                    acc += A_curr[i, j] * v[j]
                u[i] = acc

            norm_u = 0.0
            for i in range(m):
                norm_u += u[i] * u[i]
            norm_u = math.sqrt(norm_u)
            if norm_u < 1e-15:
                break
            for i in range(m):
                u[i] /= norm_u

            v_new = np.zeros((n,), dtype=np.float64)
            for j in range(n):
                acc = 0.0
                for i in range(m):
                    acc += A_curr[i, j] * u[i]
                v_new[j] = acc

            norm_v = 0.0
            for j in range(n):
                norm_v += v_new[j] * v_new[j]
            norm_v = math.sqrt(norm_v)
            if norm_v < 1e-15:
                break
            for j in range(n):
                v[j] = v_new[j] / norm_v

        sigma_vec = np.zeros((m,), dtype=np.float64)
        for i in range(m):
            acc = 0.0
            for j in range(n):
                acc += A_curr[i, j] * v[j]
            sigma_vec[i] = acc

        sigma = 0.0
        for i in range(m):
            sigma += sigma_vec[i] * sigma_vec[i]
        sigma = math.sqrt(sigma)

        S_list.append(sigma)
        U_list.append(u)
        Vt_list.append(v)

        for i in range(m):
            for j in range(n):
                A_curr[i, j] -= sigma * u[i] * v[j]

    U = np.zeros((m, k), dtype=np.float64)
    for i in range(m):
        for c in range(k):
            U[i, c] = U_list[c][i]

    S = np.zeros((k,), dtype=np.float64)
    for c in range(k):
        S[c] = S_list[c]

    Vt = np.zeros((k, n), dtype=np.float64)
    for c in range(k):
        for j in range(n):
            Vt[c, j] = Vt_list[c][j]

    return U, S, Vt


def reconstruct_svd(U: np.ndarray, S: np.ndarray, Vt: np.ndarray) -> np.ndarray:
    m = U.shape[0]
    k = U.shape[1]
    n = Vt.shape[1]

    M = np.zeros((m, k), dtype=np.float64)
    for i in range(m):
        for j in range(k):
            M[i, j] = U[i, j] * S[j]

    res = np.zeros((m, n), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            acc = 0.0
            for l in range(k):
                acc += M[i, l] * Vt[l, j]
            res[i, j] = acc

    return res
