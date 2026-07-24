import numpy as np


def broadcast_add_mul(A, B, C):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)

    shape = np.broadcast_shapes(A.shape, B.shape, C.shape)
    out = np.empty(shape, dtype=np.float64)

    for i in range(shape[0]):
        for j in range(shape[1]):
            ai = i if A.shape[0] != 1 else 0
            aj = j if A.shape[1] != 1 else 0
            bi = i if B.shape[0] != 1 else 0
            bj = j if B.shape[1] != 1 else 0
            ci = i if C.shape[0] != 1 else 0
            cj = j if C.shape[1] != 1 else 0
            out[i, j] = A[ai, aj] + B[bi, bj] * C[ci, cj]

    return out
