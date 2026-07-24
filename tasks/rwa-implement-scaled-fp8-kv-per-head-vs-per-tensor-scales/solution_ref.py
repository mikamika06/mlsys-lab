import numpy as np


def _e4m3(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    ax = np.abs(x)
    nz = ax > 0
    if np.any(nz):
        vals = np.minimum(ax[nz], 448.0)
        exp = np.maximum(np.floor(np.log2(vals)), -6)
        base = np.power(2.0, exp)
        mant = vals / base - 1.0
        mant_q = np.round(mant * 8.0) / 8.0
        vals_q = np.minimum(base * (1.0 + mant_q), 448.0)
        out[nz] = np.sign(x[nz]) * vals_q
    return out


def _qd(x, per_head):
    x = np.asarray(x, dtype=np.float64)
    if per_head:
        scale = np.max(np.abs(x), axis=(1, 2), keepdims=True) / 448.0
    else:
        scale = np.max(np.abs(x)) / 448.0
    scale = np.maximum(scale, 1e-12)
    return _e4m3(x / scale) * scale


def scaled_fp8_kv_attention(K, V, Q, per_head):
    Kd = _qd(K, per_head)
    Vd = _qd(V, per_head)
    logits = np.matmul(Q.astype(np.float64), np.swapaxes(Kd, 1, 2))
    logits = logits / np.sqrt(K.shape[-1])
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    return np.matmul(probs, Vd)
