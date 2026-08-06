import numpy as np

def act_order_perm(H):
    """Return permutation indices to sort columns by descending diagonal of H."""
    return np.argsort(-np.diag(H))

def find_damping(H, start_damp=1e-5, step=10.0, max_iter=10):
    """Find the minimum damping factor that makes Cholesky succeed."""
    diag = np.diag(H)
    mean_diag = np.mean(diag)
    for i in range(max_iter):
        damp = start_damp * (step ** i)
        H_damp = H.copy()
        np.fill_diagonal(H_damp, diag + damp * mean_diag)
        try:
            np.linalg.cholesky(H_damp)
            return damp
        except np.linalg.LinAlgError:
            pass
    raise ValueError("Cholesky failed even with max damping")

def lazy_batch_update(W, H_inv, errors, block_start, block_size):
    """Apply accumulated errors from a block of columns to the remaining columns."""
    col_end = block_start + block_size
    if col_end < W.shape[1]:
        W[:, col_end:] -= errors @ H_inv[block_start:col_end, col_end:]
    return W
