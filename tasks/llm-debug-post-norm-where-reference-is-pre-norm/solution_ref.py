import numpy as np


def _layer_norm(x, gamma, beta, eps=1e-5):
    x = np.asarray(x, dtype=np.float64)
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return ((x - mean) / np.sqrt(var + eps)) * gamma + beta


def transformer_block(x, w_attn, w_ff, gamma, beta):
    x = np.asarray(x, dtype=np.float64)
    h1 = x + _layer_norm(x, gamma, beta) @ w_attn
    return h1 + _layer_norm(h1, gamma, beta) @ w_ff
