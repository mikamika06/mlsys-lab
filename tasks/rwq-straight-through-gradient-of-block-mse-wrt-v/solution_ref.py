import numpy as np


def ste_block_mse_grad_wrt_v(X: np.ndarray, W: np.ndarray, V: np.ndarray,
                              scale: np.ndarray, bits: int) -> np.ndarray:
    """Straight-through-estimator gradient of the block MSE loss wrt V.

    See task.md for the derivation. Treats round() as identity except
    where the clip actually saturates (mask == 0 there).
    """
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    qmax = (1 << (bits - 1)) - 1

    r = V / scale[:, None]
    mask = (np.abs(r) <= qmax + 0.5).astype(np.float64)
    codes = np.clip(np.round(r), -qmax, qmax)
    Wq = scale[:, None] * codes

    B, O = X.shape[0], W.shape[0]
    pred = X @ Wq.T
    target = X @ W.T
    diff = pred - target

    return mask * (2.0 / (B * O)) * (diff.T @ X)
