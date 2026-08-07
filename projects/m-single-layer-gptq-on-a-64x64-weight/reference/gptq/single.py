import numpy as np


def run_gptq_single(W, invH, bits=4):
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
