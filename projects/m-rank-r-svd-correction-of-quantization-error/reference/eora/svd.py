import numpy as np


def compute_svd_correction(weight, q_weight, rank):
    w = np.asarray(weight, dtype=np.float32)
    qw = np.asarray(q_weight, dtype=np.float32)
    residual = w - qw
    u, s, vt = np.linalg.svd(residual, full_matrices=False)
    r = min(rank, len(s))
    if r == 0:
        return np.zeros_like(w)
    u_r = u[:, :r]
    s_r = s[:r]
    vt_r = vt[:r, :]
    correction = (u_r * s_r) @ vt_r
    return correction
