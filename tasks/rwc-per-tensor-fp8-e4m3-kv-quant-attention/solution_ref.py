import numpy as np

def _quantize_dequant(x):
    amax = np.max(np.abs(x))
    scale = amax / 448.0 if amax != 0 else 1.0
    q = np.round(x / scale)
    # Clip to the signed 8‑bit range that e4m3 can represent after rounding.
    q = np.clip(q, -127, 127).astype(np.int8)
    return q.astype(np.float32) * scale

def quantized_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    K_dq = _quantize_dequant(K)
    V_dq = _quantize_dequant(V)
    d_k = Q.shape[-1]
    scores = Q @ K_dq.T / np.sqrt(d_k)
    e = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attn = e / np.sum(e, axis=-1, keepdims=True)
    return attn @ V_dq
