import numpy as np

def compute_int8_scales(X, W):
    """Compute per-row X scales and per-column W scales for int8 quantization."""
    rows_x = X.shape[0]
    cols_x = X.shape[1]
    scale_x = np.empty(rows_x, dtype=np.float64)
    for i in range(rows_x):
        max_val = 0.0
        for j in range(cols_x):
            val = X[i, j]
            if val < 0.0:
                val = -val
            if val > max_val:
                max_val = val
        scale_x[i] = max_val / 127.0

    rows_w = W.shape[0]
    cols_w = W.shape[1]
    scale_w = np.empty(cols_w, dtype=np.float64)
    for j in range(cols_w):
        max_val = 0.0
        for i in range(rows_w):
            val = W[i, j]
            if val < 0.0:
                val = -val
            if val > max_val:
                max_val = val
        scale_w[j] = max_val / 127.0

    return scale_x, scale_w
