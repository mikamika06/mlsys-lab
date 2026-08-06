import numpy as np


def _quant_int8_pertensor(T):
    shape = T.shape
    max_val = 0.0
    for i in range(shape[0]):
        for j in range(shape[1]):
            val = float(T[i, j])
            if val < 0.0:
                val = -val
            if val > max_val:
                max_val = val
    scale = max_val / 127.0 + 1e-12

    codes = np.empty(shape, dtype=np.int8)
    for i in range(shape[0]):
        for j in range(shape[1]):
            val = float(T[i, j]) / scale
            r = round(val)
            if r < -127:
                r = -127
            elif r > 127:
                r = 127
            codes[i, j] = int(r)

    res = np.empty(shape, dtype=np.float32)
    for i in range(shape[0]):
        for j in range(shape[1]):
            res[i, j] = float(codes[i, j]) * scale
    return res


def w8a8_output_errors(X, W, s):
    """Compute W8A8 MSE for raw and SmoothQuant-smoothed quantization."""
    m = X.shape[0]
    n = X.shape[1]
    k = W.shape[1]

    Y_ref = np.empty((m, k), dtype=np.float64)
    for i in range(m):
        for j in range(k):
            acc = 0.0
            for p in range(n):
                acc += float(X[i, p]) * float(W[p, j])
            Y_ref[i, j] = acc

    X_dq = _quant_int8_pertensor(X)
    W_dq = _quant_int8_pertensor(W)

    Y_raw = np.empty((m, k), dtype=np.float64)
    for i in range(m):
        for j in range(k):
            acc = 0.0
            for p in range(n):
                acc += float(X_dq[i, p]) * float(W_dq[p, j])
            Y_raw[i, j] = acc

    mse_sum = 0.0
    count = 0
    for i in range(m):
        for j in range(k):
            diff = float(Y_raw[i, j]) - float(Y_ref[i, j])
            mse_sum += diff * diff
            count += 1
    mse_raw = mse_sum / float(count)

    X_hat = np.empty((m, n), dtype=X.dtype)
    for i in range(m):
        for j in range(n):
            X_hat[i, j] = X[i, j] / s[j]

    W_hat = np.empty((n, k), dtype=W.dtype)
    for p in range(n):
        for j in range(k):
            W_hat[p, j] = W[p, j] * s[p]

    X_hat_dq = _quant_int8_pertensor(X_hat)
    W_hat_dq = _quant_int8_pertensor(W_hat)

    Y_smooth = np.empty((m, k), dtype=np.float64)
    for i in range(m):
        for j in range(k):
            acc = 0.0
            for p in range(n):
                acc += float(X_hat_dq[i, p]) * float(W_hat_dq[p, j])
            Y_smooth[i, j] = acc

    mse_smooth_sum = 0.0
    count_smooth = 0
    for i in range(m):
        for j in range(k):
            diff = float(Y_smooth[i, j]) - float(Y_ref[i, j])
            mse_smooth_sum += diff * diff
            count_smooth += 1
    mse_smooth = mse_smooth_sum / float(count_smooth)

    return mse_raw, mse_smooth
