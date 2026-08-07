import numpy as np

def apply_low_rank_corrector(residual, rank):
    u, s, vt = np.linalg.svd(residual, full_matrices=False)
    u_r = u[:, :rank]
    s_r = s[:rank]
    vt_r = vt[:rank, :]
    approx = np.matmul(u_r * s_r, vt_r)
    cost = float(rank * (residual.shape[0] + residual.shape[1]))
    return approx, cost
