import math
import numpy as np


def _quantize_per_token_int8(x):
    """Symmetric per-row (per-token) absmax int8 quant-dequant.

    Each row gets its own scale = max(|row|) / 127. Codes are rounded to
    the nearest integer and clipped to [-127, 127], then immediately
    dequantized back to float64.
    """
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    codes = np.empty((n, d), dtype=np.float64)
    scale = np.empty((n, 1), dtype=np.float64)
    for i in range(n):
        max_val = 0.0
        for j in range(d):
            val = x[i, j]
            if val < 0.0:
                val = -val
            if val > max_val:
                max_val = val
        s = max_val / 127.0
        if s == 0.0:
            s = 1.0
        scale[i, 0] = s
        for j in range(d):
            r = round(x[i, j] / s)
            if r < -127.0:
                r = -127.0
            elif r > 127.0:
                r = 127.0
            codes[i, j] = r
    return codes * scale


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    res = np.empty((n, d), dtype=np.float64)
    for i in range(n):
        max_val = x[i, 0]
        for j in range(1, d):
            if x[i, j] > max_val:
                max_val = x[i, j]
        sum_exp = 0.0
        for j in range(d):
            val = math.exp(x[i, j] - max_val)
            res[i, j] = val
            sum_exp += val
        for j in range(d):
            res[i, j] /= sum_exp
    return res


def int8_kv_attention(Q, K, V):
    """Scaled dot-product attention with per-token INT8 quantized K and V.

    Q: (n_q, d)   K, V: (n_kv, d) / (n_kv, d_v)

    K and V are quantized row-wise (one absmax scale per key/value token)
    to int8 and immediately dequantized, then a standard
    softmax(Q K^T / sqrt(d)) V is run against the dequantized K, V. This
    mirrors a real int8 KV-cache: the quantization error only ever shows up
    through the attention math, never as a separate correction term.

    Returns
    -------
    out : np.ndarray, float64, shape (n_q, d_v)
        Attention output computed through the int8 KV path.
    mse : float
        Mean squared error between `out` and the full-precision attention
        output computed from the un-quantized K, V.
    """
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    K_hat = _quantize_per_token_int8(K)
    V_hat = _quantize_per_token_int8(V)

    d = Q.shape[1]
    n_q = Q.shape[0]
    n_kv = K.shape[0]
    d_v = V.shape[1]

    sqrt_d = math.sqrt(d)

    attn_full_raw = np.empty((n_q, n_kv), dtype=np.float64)
    for i in range(n_q):
        for j in range(n_kv):
            dot = 0.0
            for k in range(d):
                dot += Q[i, k] * K[j, k]
            attn_full_raw[i, j] = dot / sqrt_d

    full_softmax = _softmax(attn_full_raw)

    full = np.empty((n_q, d_v), dtype=np.float64)
    for i in range(n_q):
        for j in range(d_v):
            dot = 0.0
            for k in range(n_kv):
                dot += full_softmax[i, k] * V[k, j]
            full[i, j] = dot

    attn_hat_raw = np.empty((n_q, n_kv), dtype=np.float64)
    for i in range(n_q):
        for j in range(n_kv):
            dot = 0.0
            for k in range(d):
                dot += Q[i, k] * K_hat[j, k]
            attn_hat_raw[i, j] = dot / sqrt_d

    hat_softmax = _softmax(attn_hat_raw)

    out = np.empty((n_q, d_v), dtype=np.float64)
    for i in range(n_q):
        for j in range(d_v):
            dot = 0.0
            for k in range(n_kv):
                dot += hat_softmax[i, k] * V_hat[k, j]
            out[i, j] = dot

    sum_sq_err = 0.0
    count = 0
    for i in range(n_q):
        for j in range(d_v):
            diff = out[i, j] - full[i, j]
            sum_sq_err += diff * diff
            count += 1
    mse = float(sum_sq_err / count)

    return out, mse
