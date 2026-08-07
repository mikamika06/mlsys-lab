import numpy as np


def init_loftq_residual(W, rank):
    W_q = np.round(W * 4.0) / 4.0
    residual = W - W_q
    U, S, Vt = np.linalg.svd(residual, full_matrices=False)
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vt_r = Vt[:rank, :]
    sqrt_S = np.sqrt(S_r)
    A = np.diag(sqrt_S) @ Vt_r
    B = U_r @ np.diag(sqrt_S)
    return W_q, A, B
