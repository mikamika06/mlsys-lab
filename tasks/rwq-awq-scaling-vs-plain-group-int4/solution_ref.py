import numpy as np


def _quantize_group_int4(W, group_size):
    W = np.asarray(W, dtype=np.float64)
    out = np.empty_like(W)
    for start in range(0, W.shape[1], group_size):
        end = min(start + group_size, W.shape[1])
        group = W[:, start:end]
        scale = max(float(np.max(np.abs(group))) / 7.0, 1e-12)
        out[:, start:end] = np.clip(np.round(group / scale), -8, 7) * scale
    return out


def awq_vs_plain_group_int4_mse(W, X, group_size):
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)

    plain = _quantize_group_int4(W, group_size)

    importance = np.mean(np.abs(X), axis=0)
    channel_scale = (importance / (np.mean(importance) + 1e-12)) ** 0.5

    scaled = W * channel_scale
    awq = _quantize_group_int4(scaled, group_size) / channel_scale

    y = X @ W.T
    awq_mse = float(np.mean((y - X @ awq.T) ** 2))
    plain_mse = float(np.mean((y - X @ plain.T) ** 2))

    return awq_mse, plain_mse
