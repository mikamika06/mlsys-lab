import numpy as np


def get_scale_zp(W, bits=4):
    qmax = (1 << bits) - 1
    w_min = np.min(W, axis=-1, keepdims=True)
    w_max = np.max(W, axis=-1, keepdims=True)
    scale = np.maximum((w_max - w_min) / qmax, 1e-8)
    zp = np.round(-w_min / scale)
    return scale, zp


def rtn_quantize(W, bits=4):
    scale, zp = get_scale_zp(W, bits)
    qmax = (1 << bits) - 1
    q_int = np.clip(np.round(W / scale + zp), 0, qmax)
    W_q = (q_int - zp) * scale
    return W_q, scale, zp


def learned_round_layer(W, X, steps=100, lr=0.1, bits=4):
    scale, zp = get_scale_zp(W, bits)
    qmax = (1 << bits) - 1
    W_floor = np.floor(W / scale + zp)
    V = np.zeros_like(W)
    Y_fp = X @ W.T
    N = X.shape[0]

    for _ in range(steps):
        sig_V = 1.0 / (1.0 + np.exp(-V))
        q_continuous = np.clip(W_floor + sig_V, 0, qmax)
        W_q_soft = (q_continuous - zp) * scale
        Y_q = X @ W_q_soft.T
        diff = Y_q - Y_fp
        dL_dWq = (2.0 / (N * W.shape[0])) * (diff.T @ X)
        dL_dV = dL_dWq * scale * sig_V * (1.0 - sig_V)
        V = V - lr * np.sign(dL_dV)

    sig_V_final = 1.0 / (1.0 + np.exp(-V))
    q_final = np.clip(W_floor + (sig_V_final >= 0.5).astype(float), 0, qmax)
    W_q_opt = (q_final - zp) * scale
    return W_q_opt, scale, zp
