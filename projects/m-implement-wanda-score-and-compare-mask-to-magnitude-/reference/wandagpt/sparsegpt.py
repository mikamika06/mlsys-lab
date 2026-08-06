import numpy as np


def simplified_sparsegpt(
    W: np.ndarray, X: np.ndarray, sparsity: float, damping: float = 1e-4
) -> np.ndarray:
    """Perform single-layer Hessian OBS weight updates given activations X."""
    W_out = W.copy().astype(np.float64)
    d_in = W.shape[1]

    H = 2.0 * (X @ X.T)
    diag_mean = np.mean(np.diag(H))
    H += damping * diag_mean * np.eye(d_in)

    H_inv = np.linalg.inv(H)

    scores = (W_out ** 2) / (np.diag(H_inv)[None, :] + 1e-10)
    k = int(np.round(W.size * sparsity))

    if k == 0:
        return W_out

    flat_indices = np.argsort(scores.ravel())[:k]

    mask = np.ones_like(W_out, dtype=bool)
    for idx in flat_indices:
        r, c = np.unravel_index(idx, W_out.shape)
        mask[r, c] = False

    for c in range(d_in):
        pruned_rows = np.where(~mask[:, c])[0]
        if len(pruned_rows) == 0:
            continue
        h_inv_cc = H_inv[c, c]
        if abs(h_inv_cc) < 1e-12:
            continue
        delta_w = W_out[:, c:c+1] / h_inv_cc
        W_out -= delta_w @ H_inv[c:c+1, :]
        W_out[pruned_rows, c] = 0.0

    W_out[~mask] = 0.0
    return W_out
