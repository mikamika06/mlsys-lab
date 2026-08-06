import math
import numpy as np


def _quantize_rows(x):
    x = np.asarray(x, dtype=np.float64)
    n_rows, n_cols = x.shape
    out = np.empty((n_rows, n_cols), dtype=np.float64)
    for i in range(n_rows):
        max_abs = 0.0
        for j in range(n_cols):
            val = abs(x[i, j])
            if val > max_abs:
                max_abs = val
        scale = max_abs / 127.0
        if scale == 0.0:
            scale = 1.0
        for j in range(n_cols):
            scaled_val = x[i, j] / scale
            rounded = round(scaled_val)
            if rounded < -127:
                clipped = -127
            elif rounded > 127:
                clipped = 127
            else:
                clipped = rounded
            out[i, j] = float(int(clipped)) * scale
    return out


def _softmax(x):
    n_rows, n_cols = x.shape
    out = np.empty((n_rows, n_cols), dtype=np.float64)
    for i in range(n_rows):
        max_val = x[i, 0]
        for j in range(1, n_cols):
            if x[i, j] > max_val:
                max_val = x[i, j]
        sum_exp = 0.0
        for j in range(n_cols):
            e_val = math.exp(x[i, j] - max_val)
            out[i, j] = e_val
            sum_exp += e_val
        for j in range(n_cols):
            out[i, j] /= sum_exp
    return out


def quantized_kv_attention(Q, K, V):
    Q = np.asarray(Q, dtype=np.float64)
    K_hat = _quantize_rows(K)
    V_hat = _quantize_rows(V)

    n_q, d = Q.shape
    n_k, d_k = K_hat.shape
    n_v, d_v = V_hat.shape

    sqrt_d = math.sqrt(d)
    scores = np.empty((n_q, n_k), dtype=np.float64)
    for i in range(n_q):
        for j in range(n_k):
            dot_val = 0.0
            for k in range(d):
                dot_val += Q[i, k] * K_hat[j, k]
            scores[i, j] = dot_val / sqrt_d

    attn_weights = _softmax(scores)

    out = np.empty((n_q, d_v), dtype=np.float64)
    for i in range(n_q):
        for j in range(d_v):
            dot_val = 0.0
            for k in range(n_k):
                dot_val += attn_weights[i, k] * V_hat[k, j]
            out[i, j] = dot_val

    return out.astype(np.float64)
