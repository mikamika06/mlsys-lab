import math
import numpy as np


def householder_qr_reconstruct(A):
    A = np.asarray(A, dtype=np.float64)
    m, n = A.shape
    R = A.copy()
    Q = np.eye(m, dtype=np.float64)

    for k in range(min(m, n)):
        x = R[k:, k].copy()
        norm_sq = 0.0
        for val in x:
            norm_sq += val * val
        norm = math.sqrt(norm_sq)
        if norm == 0:
            continue

        sign = -1.0 if x[0] < 0 else 1.0
        v = x.copy()
        v[0] += sign * norm

        vnorm_sq = 0.0
        for val in v:
            vnorm_sq += val * val
        vnorm = math.sqrt(vnorm_sq)
        if vnorm == 0:
            continue
        for i in range(len(v)):
            v[i] /= vnorm

        sub_R = R[k:, k:]
        dot_res = []
        for j in range(sub_R.shape[1]):
            s = 0.0
            for i in range(len(v)):
                s += v[i] * sub_R[i, j]
            dot_res.append(s)

        for i in range(len(v)):
            for j in range(sub_R.shape[1]):
                sub_R[i, j] -= 2.0 * v[i] * dot_res[j]

        sub_Q = Q[:, k:]
        matvec = []
        for i in range(sub_Q.shape[0]):
            s = 0.0
            for j in range(len(v)):
                s += sub_Q[i, j] * v[j]
            matvec.append(s)

        for i in range(sub_Q.shape[0]):
            for j in range(len(v)):
                sub_Q[i, j] -= 2.0 * matvec[i] * v[j]

    return Q, R
