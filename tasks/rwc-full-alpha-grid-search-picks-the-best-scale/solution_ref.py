import numpy as np


def _quantize_rows(x):
    max_abs = np.max(np.abs(x), axis=1, keepdims=True)
    scale = max_abs / 127.0
    q = np.round(x / scale)
    return q * scale


def search_awq_alpha(W, X, s_x):
    alphas = np.arange(20, dtype=np.float64) / 20.0
    target = W @ X
    losses = []

    for alpha in alphas:
        s = np.power(s_x, alpha)
        scaled = W * s[np.newaxis, :]
        quantized = _quantize_rows(scaled)
        restored = quantized * (1.0 / s)[np.newaxis, :]
        losses.append(np.linalg.norm(target - restored @ X))

    losses = np.asarray(losses, dtype=np.float64)
    return int(np.argmin(losses)), losses
