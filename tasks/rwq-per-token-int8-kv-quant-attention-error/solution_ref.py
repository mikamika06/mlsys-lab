import numpy as np


def _quantize_per_token_int8(x):
    """Symmetric per-row (per-token) absmax int8 quant-dequant.

    Each row gets its own scale = max(|row|) / 127. Codes are rounded to
    the nearest integer and clipped to [-127, 127], then immediately
    dequantized back to float64.
    """
    x = np.asarray(x, dtype=np.float64)
    scale = np.max(np.abs(x), axis=1, keepdims=True) / 127.0
    scale = np.where(scale == 0, 1.0, scale)
    codes = np.clip(np.round(x / scale), -127, 127)
    return codes * scale


def _softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=1, keepdims=True)


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

    scale = np.sqrt(Q.shape[1])
    full = _softmax(Q @ K.T / scale) @ V
    out = _softmax(Q @ K_hat.T / scale) @ V_hat

    mse = float(np.mean((out - full) ** 2))
    return out, mse
