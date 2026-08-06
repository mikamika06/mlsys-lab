import math
import numpy as np


def _quantize_group_int4(W, group_size):
    W = np.asarray(W, dtype=np.float64)
    rows, cols = W.shape
    out = np.empty_like(W)

    for start in range(0, cols, group_size):
        end = min(start + group_size, cols)

        max_val = 0.0
        for r in range(rows):
            for c in range(start, end):
                val = abs(W[r, c])
                if val > max_val:
                    max_val = val

        scale = max_val / 7.0
        if scale < 1e-12:
            scale = 1e-12

        for r in range(rows):
            for c in range(start, end):
                q = round(W[r, c] / scale)
                if q < -8:
                    q = -8
                elif q > 7:
                    q = 7
                out[r, c] = q * scale

    return out


def awq_vs_plain_group_int4_mse(W, X, group_size):
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)

    rows_w, cols_w = W.shape
    rows_x, cols_x = X.shape

    plain = _quantize_group_int4(W, group_size)

    importance = np.empty(cols_x, dtype=np.float64)
    for j in range(cols_x):
        col_sum = 0.0
        for i in range(rows_x):
            col_sum += abs(X[i, j])
        importance[j] = col_sum / float(rows_x)

    imp_sum = 0.0
    for j in range(cols_x):
        imp_sum += importance[j]
    mean_imp = imp_sum / float(cols_x)

    channel_scale = np.empty(cols_x, dtype=np.float64)
    denom = mean_imp + 1e-12
    for j in range(cols_x):
        channel_scale[j] = math.sqrt(importance[j] / denom)

    scaled = np.empty_like(W)
    for i in range(rows_w):
        for j in range(cols_w):
            scaled[i, j] = W[i, j] * channel_scale[j]

    awq_quant = _quantize_group_int4(scaled, group_size)
    awq = np.empty_like(W)
    for i in range(rows_w):
        for j in range(cols_w):
            awq[i, j] = awq_quant[i, j] / channel_scale[j]

    y = np.empty((rows_x, rows_w), dtype=np.float64)
    for i in range(rows_x):
        for j in range(rows_w):
            acc = 0.0
            for k in range(cols_x):
                acc += X[i, k] * W[j, k]
            y[i, j] = acc

    awq_mse_sum = 0.0
    for i in range(rows_x):
        for j in range(rows_w):
            acc = 0.0
            for k in range(cols_x):
                acc += X[i, k] * awq[j, k]
            diff = y[i, j] - acc
            awq_mse_sum += diff * diff
    awq_mse = float(awq_mse_sum / float(rows_x * rows_w))

    plain_mse_sum = 0.0
    for i in range(rows_x):
        for j in range(rows_w):
            acc = 0.0
            for k in range(cols_x):
                acc += X[i, k] * plain[j, k]
            diff = y[i, j] - acc
            plain_mse_sum += diff * diff
    plain_mse = float(plain_mse_sum / float(rows_x * rows_w))

    return awq_mse, plain_mse
