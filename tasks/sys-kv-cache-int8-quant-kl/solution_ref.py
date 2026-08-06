import math
import numpy as np


def _quantize_int8_per_token(a: np.ndarray):
    """Symmetric per-row (per-token) int8 quantization.

    a : (n, d) float array -- one row per cached token.
    Returns (codes, scale): codes is (n, d) int8, scale is (n, 1) float32
    such that ``codes[i].astype(float32) * scale[i] ~= a[i]``.
    """
    n = a.shape[0]
    d = a.shape[1]
    scale_list = []
    for i in range(n):
        max_val = 0.0
        for j in range(d):
            val = a[i, j]
            if val < 0:
                val = -val
            if val > max_val:
                max_val = val
        s = max_val / 127.0
        if s == 0.0:
            s = 1.0
        scale_list.append([s])
    scale = np.array(scale_list, dtype=np.float32)

    codes_list = []
    for i in range(n):
        row_codes = []
        s = scale[i, 0]
        for j in range(d):
            r = round(a[i, j] / s)
            if r < -127:
                r = -127
            elif r > 127:
                r = 127
            row_codes.append(int(r))
        codes_list.append(row_codes)
    codes = np.array(codes_list, dtype=np.int8)

    return codes, scale


def _dequantize_int8(codes: np.ndarray, scale: np.ndarray) -> np.ndarray:
    n = codes.shape[0]
    d = codes.shape[1]
    res_list = []
    for i in range(n):
        row = []
        s = scale[i, 0]
        for j in range(d):
            row.append(float(codes[i, j]) * s)
        res_list.append(row)
    return np.array(res_list, dtype=np.float32)


def kv_cache_int8_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray):
    """Single-head scaled dot-product attention against an int8-quantized
    KV cache.

    Q : (m, d) fp32 queries.
    K, V : (n, d) fp32 keys/values, as they would be written into the KV
        cache.

    Both K and V are quantized to int8 with a PER-TOKEN (per-row)
    symmetric scale before being used -- this is what an int8 KV-cache
    actually stores. Attention is then computed against the dequantized
    cache:

        logits  = (Q @ K_hat.T) / sqrt(d)
        weights = softmax(logits, axis=-1)
        out     = weights @ V_hat

    Returns
    -------
    logits : (m, n) float64
        Pre-softmax attention scores computed from the dequantized K.
    out : (m, d) float64
        The attention output computed from the dequantized K and V.
    """
    Q = np.asarray(Q, dtype=np.float32)
    K = np.asarray(K, dtype=np.float32)
    V = np.asarray(V, dtype=np.float32)
    d = Q.shape[-1]

    K_codes, K_scale = _quantize_int8_per_token(K)
    V_codes, V_scale = _quantize_int8_per_token(V)
    K_hat = _dequantize_int8(K_codes, K_scale)
    V_hat = _dequantize_int8(V_codes, V_scale)

    m = Q.shape[0]
    n = K_hat.shape[0]
    sqrt_d = math.sqrt(d)

    logits_list = []
    for i in range(m):
        row = []
        for j in range(n):
            acc = 0.0
            for k_idx in range(d):
                acc += float(Q[i, k_idx]) * float(K_hat[j, k_idx])
            row.append(acc / sqrt_d)
        logits_list.append(row)
    logits = np.array(logits_list, dtype=np.float64)

    z_list = []
    for i in range(m):
        max_val = logits[i, 0]
        for j in range(1, n):
            val = logits[i, j]
            if val > max_val:
                max_val = val
        row = []
        for j in range(n):
            row.append(logits[i, j] - max_val)
        z_list.append(row)

    w_list = []
    for i in range(m):
        row_exp = []
        sum_exp = 0.0
        for j in range(n):
            e = math.exp(z_list[i][j])
            row_exp.append(e)
            sum_exp += e
        row_w = []
        for j in range(n):
            row_w.append(row_exp[j] / sum_exp)
        w_list.append(row_w)

    v_d = V_hat.shape[1]
    out_list = []
    for i in range(m):
        row = []
        for j in range(v_d):
            acc = 0.0
            for k_idx in range(n):
                acc += w_list[i][k_idx] * float(V_hat[k_idx, j])
            row.append(acc)
        out_list.append(row)
    out = np.array(out_list, dtype=np.float64)

    return logits, out
