import numpy as np


def _int8_symmetric_roundtrip(x: np.ndarray) -> np.ndarray:
    """Per-tensor symmetric INT8 quantize-then-dequantize."""
    amax = float(np.max(np.abs(x)))
    scale = max(amax / 127.0, 1e-12)
    q = np.clip(np.round(x / scale), -127, 127)
    return q * scale


def smoothquant_w8a8_comparison(X: np.ndarray, W: np.ndarray, alpha: float) -> dict:
    """Compare per-tensor symmetric INT8 W8A8 (weight + activation)
    quantization error for a linear layer Y = X @ W^T, with and without
    SmoothQuant's activation-outlier migration.

    X : (n, d_in) activations (outlier-heavy input channels).
    W : (d_out, d_in) weight matrix.
    alpha : float in (0, 1), SmoothQuant's migration strength.

    SmoothQuant computes a PER-INPUT-CHANNEL scale
        s_j = max_i(|X[i,j]|)^alpha / max_o(|W[o,j]|)^(1-alpha)
    and rescales X_smooth[:, j] = X[:, j] / s_j, W_smooth[:, j] = W[:, j] * s_j
    -- mathematically X_smooth @ W_smooth^T == X @ W^T exactly (no
    approximation), it only redistributes dynamic range from activations
    (hard to quantize, outlier-heavy) onto weights (easy to quantize,
    already well-behaved) BEFORE both are cast to INT8.

    Returns a dict:
      "error_raw"        : Frobenius relative error of the raw (no
                            smoothing) W8A8 output vs the exact FP output.
      "error_smoothed"    : same, with SmoothQuant migration applied
                            first.
      "improvement_ratio" : error_smoothed / error_raw (< 1 means
                            smoothing helped).
    """
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)

    Y_true = X @ W.T

    # -- raw W8A8 --
    X_hat_raw = _int8_symmetric_roundtrip(X)
    W_hat_raw = _int8_symmetric_roundtrip(W)
    Y_raw = X_hat_raw @ W_hat_raw.T
    error_raw = float(np.linalg.norm(Y_raw - Y_true) / np.linalg.norm(Y_true))

    # -- SmoothQuant migration, then W8A8 --
    x_amax = np.max(np.abs(X), axis=0)   # (d_in,)
    w_amax = np.max(np.abs(W), axis=0)   # (d_in,)
    s = (x_amax ** alpha) / np.maximum(w_amax ** (1.0 - alpha), 1e-12)
    s = np.maximum(s, 1e-12)

    X_smooth = X / s[None, :]
    W_smooth = W * s[None, :]

    X_hat_sm = _int8_symmetric_roundtrip(X_smooth)
    W_hat_sm = _int8_symmetric_roundtrip(W_smooth)
    Y_smooth = X_hat_sm @ W_hat_sm.T
    error_smoothed = float(np.linalg.norm(Y_smooth - Y_true) / np.linalg.norm(Y_true))

    improvement_ratio = error_smoothed / error_raw if error_raw > 0 else float("inf")

    return {
        "error_raw": error_raw,
        "error_smoothed": error_smoothed,
        "improvement_ratio": improvement_ratio,
    }
