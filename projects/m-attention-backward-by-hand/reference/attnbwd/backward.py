import numpy as np


def generate_dropout_mask(shape, p, seed):
    if p <= 0.0:
        return np.ones(shape, dtype=np.float64)
    rng = np.random.RandomState(seed)
    return (rng.rand(*shape) >= p).astype(np.float64)


def attention_backward(Q, K, V, dO, p=0.0, seed=0, scale=None):
    B, H, N, D = Q.shape
    if scale is None:
        scale = 1.0 / np.sqrt(D)
    S = scale * np.matmul(Q, K.swapaxes(-1, -2))
    S_max = np.max(S, axis=-1, keepdims=True)
    exp_S = np.exp(S - S_max)
    P_raw = exp_S / np.sum(exp_S, axis=-1, keepdims=True)
    mask = generate_dropout_mask(S.shape, p, seed)
    scale_drop = 1.0 / (1.0 - p) if p < 1.0 else 0.0
    P_drop = P_raw * mask * scale_drop
    dV = np.matmul(P_drop.swapaxes(-1, -2), dO)
    dP_drop = np.matmul(dO, V.swapaxes(-1, -2))
    dP_raw = dP_drop * mask * scale_drop
    D_sum = np.sum(dP_raw * P_raw, axis=-1, keepdims=True)
    dS = scale * P_raw * (dP_raw - D_sum)
    dQ = np.matmul(dS, K)
    dK = np.matmul(dS.swapaxes(-1, -2), Q)
    return dQ, dK, dV
