import numpy as np

EPS = 1e-8


def _w8a8_mse(W, X, alpha):
    amax_x = np.max(np.abs(X), axis=0)
    amax_w = np.max(np.abs(W), axis=0)
    s = np.power(amax_x, alpha) / (np.power(amax_w, 1.0 - alpha) + EPS)
    s = np.maximum(s, EPS)

    Xs = X / s[None, :]
    Ws = W * s[None, :]

    sx = np.max(np.abs(Xs))
    sx = sx / 127.0 if sx > 0 else 1.0
    Xq = np.clip(np.round(Xs / sx), -127, 127) * sx

    aw = np.max(np.abs(Ws), axis=1)
    sw = np.where(aw > 0, aw / 127.0, 1.0)
    Wq = np.clip(np.round(Ws / sw[:, None]), -127, 127) * sw[:, None]

    Yhat = Xq @ Wq.T
    Y = X @ W.T
    return float(np.mean((Yhat - Y) ** 2))


def sweep_alpha(W: np.ndarray, X: np.ndarray, alphas: np.ndarray):
    """
    For each alpha in `alphas`:
      1. Per-input-channel smoothing scale
         s_j = max(|X[:,j]|)^alpha / max(|W[:,j]|)^(1-alpha).
      2. Smooth: X' = X / s, W' = W * s.
      3. Quantize X' to int8 with one dynamic per-tensor scale
         (max(|X'|)/127); quantize W' to int8 with one scale per output
         row (max(|W'[row]|)/127). Dequantize both.
      4. Output MSE = mean((X'_deq @ W'_deq^T - X @ W^T)^2).

    Returns (best_idx, best_mse): the index into `alphas` achieving the
    lowest output MSE, and that MSE.
    """
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    mses = [_w8a8_mse(W, X, float(a)) for a in alphas]
    idx = int(np.argmin(mses))
    return idx, mses[idx]
