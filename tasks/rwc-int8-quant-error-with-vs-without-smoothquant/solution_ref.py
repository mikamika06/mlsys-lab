import math
import numpy as np


def _int8_symmetric_roundtrip(x: np.ndarray) -> np.ndarray:
    """Per-tensor symmetric INT8 quantize-then-dequantize."""
    amax = 0.0
    shape = x.shape
    if len(shape) == 1:
        for i in range(shape[0]):
            val = abs(x[i])
            if val > amax:
                amax = val
        scale = max(amax / 127.0, 1e-12)
        res = np.zeros(shape, dtype=np.float64)
        for i in range(shape[0]):
            q = max(-127, min(127, round(x[i] / scale)))
            res[i] = q * scale
        return res
    else:
        for i in range(shape[0]):
            for j in range(shape[1]):
                val = abs(x[i, j])
                if val > amax:
                    amax = val
        scale = max(amax / 127.0, 1e-12)
        res = np.zeros(shape, dtype=np.float64)
        for i in range(shape[0]):
            for j in range(shape[1]):
                q = max(-127, min(127, round(x[i, j] / scale)))
                res[i, j] = q * scale
        return res


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

    n, d_in = X.shape
    d_out, _ = W.shape

    Y_true = np.zeros((n, d_out), dtype=np.float64)
    for i in range(n):
        for j in range(d_out):
            acc = 0.0
            for k in range(d_in):
                acc += X[i, k] * W[j, k]
            Y_true[i, j] = acc

    X_hat_raw = _int8_symmetric_roundtrip(X)
    W_hat_raw = _int8_symmetric_roundtrip(W)
    
    Y_raw = np.zeros((n, d_out), dtype=np.float64)
    for i in range(n):
        for j in range(d_out):
            acc = 0.0
            for k in range(d_in):
                acc += X_hat_raw[i, k] * W_hat_raw[j, k]
            Y_raw[i, j] = acc

    sum_sq_raw = 0.0
    sum_sq_true = 0.0
    for i in range(n):
        for j in range(d_out):
            diff = Y_raw[i, j] - Y_true[i, j]
            sum_sq_raw += diff * diff
            val = Y_true[i, j]
            sum_sq_true += val * val
    error_raw = float(math.sqrt(sum_sq_raw) / math.sqrt(sum_sq_true))

    x_amax = [0.0] * d_in
    for j in range(d_in):
        col_max = 0.0
        for i in range(n):
            val = abs(X[i, j])
            if val > col_max:
                col_max = val
        x_amax[j] = col_max

    w_amax = [0.0] * d_in
    for j in range(d_in):
        col_max = 0.0
        for i in range(d_out):
            val = abs(W[i, j])
            if val > col_max:
                col_max = val
        w_amax[j] = col_max

    s = [0.0] * d_in
    for j in range(d_in):
        num = x_amax[j] ** alpha
        denom = max(w_amax[j] ** (1.0 - alpha), 1e-12)
        s[j] = max(num / denom, 1e-12)

    X_smooth = np.zeros((n, d_in), dtype=np.float64)
    for i in range(n):
        for j in range(d_in):
            X_smooth[i, j] = X[i, j] / s[j]

    W_smooth = np.zeros((d_out, d_in), dtype=np.float64)
    for i in range(d_out):
        for j in range(d_in):
            W_smooth[i, j] = W[i, j] * s[j]

    X_hat_sm = _int8_symmetric_roundtrip(X_smooth)
    W_hat_sm = _int8_symmetric_roundtrip(W_smooth)
    
    Y_smooth = np.zeros((n, d_out), dtype=np.float64)
    for i in range(n):
        for j in range(d_out):
            acc = 0.0
            for k in range(d_in):
                acc += X_hat_sm[i, k] * W_hat_sm[j, k]
            Y_smooth[i, j] = acc

    sum_sq_sm = 0.0
    for i in range(n):
        for j in range(d_out):
            diff = Y_smooth[i, j] - Y_true[i, j]
            sum_sq_sm += diff * diff
    error_smoothed = float(math.sqrt(sum_sq_sm) / math.sqrt(sum_sq_true))

    improvement_ratio = error_smoothed / error_raw if error_raw > 0 else float("inf")

    return {
        "error_raw": error_raw,
        "error_smoothed": error_smoothed,
        "improvement_ratio": improvement_ratio,
    }
