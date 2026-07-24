import numpy as np


def _fp8_e4m3_round(x):
    x = np.asarray(x, dtype=np.float64)
    sign = np.sign(x)
    ax = np.abs(x)
    out = np.zeros_like(ax)

    normal = ax >= 2 ** -6
    if np.any(normal):
        a = ax[normal]
        exp = np.floor(np.log2(a))
        exp = np.clip(exp, -6, 8)
        step = 2.0 ** (exp - 3)
        out[normal] = np.minimum(np.round(a / step) * step, 448.0)

    sub = ~normal
    if np.any(sub):
        out[sub] = np.round(ax[sub] / (2 ** -9)) * (2 ** -9)

    return sign * out


def _quantize_per_token(x):
    scale = np.max(np.abs(x), axis=1, keepdims=True) / 448.0
    scale = np.where(scale == 0, 1.0, scale)
    return _fp8_e4m3_round(x / scale) * scale


def _softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=1, keepdims=True)


def fp8_kv_attention(Q, K, V):
    K_hat = _quantize_per_token(K)
    V_hat = _quantize_per_token(V)

    full = _softmax(Q @ K.T / np.sqrt(Q.shape[1])) @ V
    out = _softmax(Q @ K_hat.T / np.sqrt(Q.shape[1])) @ V_hat
    mse = float(np.mean((out - full) ** 2))
    return out.astype(np.float64), mse
