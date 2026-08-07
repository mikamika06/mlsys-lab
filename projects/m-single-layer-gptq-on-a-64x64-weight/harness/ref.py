import numpy as np


def generate_inputs(seed=42):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((64, 64))
    X = rng.standard_normal((128, 64))
    H = X.T @ X + 1e-4 * np.eye(64)
    invH = np.linalg.inv(H)
    return W, invH, X


def reference_obq_update(w_col, invH_col_idx, invH_val):
    return w_col - (w_col / invH_val) * invH_col_idx


def reference_gptq_single(W, invH, bits=4):
    W_q = W.copy()
    Hinv = invH.copy()
    d = W.shape[1]
    qmin = -(2 ** (bits - 1))
    qmax = (2 ** (bits - 1)) - 1
    scale = np.max(np.abs(W), axis=0) / qmax
    scale = np.maximum(scale, 1e-8)
    for i in range(d):
        w = W_q[:, i]
        hinv_val = Hinv[i, i]
        q = np.round(w / scale[i])
        q = np.clip(q, qmin, qmax)
        w_quant = q * scale[i]
        err = (w - w_quant) / hinv_val
        W_q[:, i:] -= np.outer(err, Hinv[i, i:])
        W_q[:, i] = w_quant
    return W_q
