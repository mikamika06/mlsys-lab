import numpy as np


def wanda_score_mask(W: np.ndarray, X: np.ndarray, sparsity: float) -> np.ndarray:
    col_norm = np.linalg.norm(X, axis=0)
    S = np.abs(W) * col_norm[None, :]
    d_out, d_in = W.shape
    k = max(1, int(round((1.0 - sparsity) * d_in)))

    order = np.argsort(-S, axis=1, kind="stable")
    mask = np.zeros((d_out, d_in), dtype=bool)
    rows_idx = np.arange(d_out)[:, None]
    mask[rows_idx, order[:, :k]] = True
    return mask
