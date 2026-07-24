import numpy as np


def _quantize(x, kind):
    if kind == 0:
        scale = np.max(np.abs(x)) / 127.0
        return np.clip(np.round(x / scale), -127, 127) * scale, np.array([scale])
    if kind == 1:
        scale = np.max(np.abs(x), axis=(0, 2), keepdims=True) / 127.0
        return np.clip(np.round(x / scale), -127, 127) * scale, scale.reshape(-1)
    scale = np.max(np.abs(x), axis=(1, 2), keepdims=True) / 127.0
    return np.clip(np.round(x / scale), -127, 127) * scale, scale.reshape(-1)


def _attention(Q, K, V):
    scores = np.matmul(Q, np.transpose(K, (0, 2, 1))) / np.sqrt(K.shape[-1])
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs /= np.sum(probs, axis=-1, keepdims=True)
    return np.matmul(probs, V)


def choose_kv_scale_granularity(K, V, Q, budget):
    ref = _attention(Q, K, V)
    best_cost = None
    best_idx = 0
    for i in range(3):
        Kq, ks = _quantize(K, i)
        Vq, vs = _quantize(V, i)
        out = _attention(Q, Kq, Vq)
        mse = np.mean((ref - out) ** 2)
        scale_bytes = (ks.size + vs.size) * 4
        cost = float(mse + 0.001 * max(0, scale_bytes - budget))
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_idx = i
    return best_idx
