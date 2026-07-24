import numpy as np


def _quantize_int8_per_token(a: np.ndarray):
    """Symmetric per-row (per-token) int8 quantization.

    a : (n, d) float array -- one row per cached token.
    Returns (codes, scale): codes is (n, d) int8, scale is (n, 1) float32
    such that ``codes[i].astype(float32) * scale[i] ~= a[i]``.
    """
    scale = np.max(np.abs(a), axis=-1, keepdims=True) / 127.0
    scale = np.where(scale == 0, 1.0, scale).astype(np.float32)
    codes = np.clip(np.round(a / scale), -127, 127).astype(np.int8)
    return codes, scale


def _dequantize_int8(codes: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return codes.astype(np.float32) * scale


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

    logits = (Q.astype(np.float64) @ K_hat.astype(np.float64).T) / np.sqrt(d)
    z = logits - np.max(logits, axis=-1, keepdims=True)
    w = np.exp(z)
    w = w / np.sum(w, axis=-1, keepdims=True)
    out = w @ V_hat.astype(np.float64)

    return logits, out
