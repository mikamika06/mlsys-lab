import numpy as np


def _sym_quant_per_row(A: np.ndarray) -> np.ndarray:
    rows = A.shape[0]
    cols = A.shape[1]
    res = np.zeros((rows, cols), dtype=np.float64)
    for i in range(rows):
        m = 0.0
        for j in range(cols):
            val = abs(float(A[i, j]))
            if val > m:
                m = val
        if m == 0.0:
            m = 1.0
        scale = m / 127.0
        for j in range(cols):
            divided = float(A[i, j]) / scale
            rounded = round(divided)
            clipped = max(-127.0, min(127.0, rounded))
            res[i, j] = clipped * scale
    return res


def weight_only_vs_dynamic_mse(x: np.ndarray, W: np.ndarray):
    """
    Compute a linear layer's output MSE two ways vs the full-precision
    reference y_fp = x @ W.T:

    - weight-only int8: quantize only W (per-output-row symmetric int8),
      keep x in full precision.
    - int8-dynamic: additionally quantize x per-row (per-sample,
      symmetric, recomputed on the fly from the batch) before the matmul.

    Returns (mse_weight_only, mse_dynamic).
    """
    x = np.asarray(x, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)

    b = x.shape[0]
    d_in = x.shape[1]
    d_out = W.shape[0]

    y_fp = np.zeros((b, d_out), dtype=np.float64)
    for i in range(b):
        for j in range(d_out):
            s = 0.0
            for k in range(d_in):
                s += x[i, k] * W[j, k]
            y_fp[i, j] = s

    W_hat = _sym_quant_per_row(W)

    y_wo = np.zeros((b, d_out), dtype=np.float64)
    for i in range(b):
        for j in range(d_out):
            s = 0.0
            for k in range(d_in):
                s += x[i, k] * W_hat[j, k]
            y_wo[i, j] = s

    sum_sq_wo = 0.0
    for i in range(b):
        for j in range(d_out):
            diff = y_fp[i, j] - y_wo[i, j]
            sum_sq_wo += diff * diff
    mse_wo = float(sum_sq_wo / (b * d_out))

    x_hat = _sym_quant_per_row(x)

    y_dyn = np.zeros((b, d_out), dtype=np.float64)
    for i in range(b):
        for j in range(d_out):
            s = 0.0
            for k in range(d_in):
                s += x_hat[i, k] * W_hat[j, k]
            y_dyn[i, j] = s

    sum_sq_dyn = 0.0
    for i in range(b):
        for j in range(d_out):
            diff = y_fp[i, j] - y_dyn[i, j]
            sum_sq_dyn += diff * diff
    mse_dyn = float(sum_sq_dyn / (b * d_out))

    return mse_wo, mse_dyn
