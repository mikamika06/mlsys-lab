"""Single layer GPTQ implementation."""
import numpy as np

def round_to_nearest(weights, bits=4):
    w_min = weights.min(axis=0, keepdims=True)
    w_max = weights.max(axis=0, keepdims=True)
    levels = (1 << bits) - 1
    scale = np.maximum((w_max - w_min) / levels, 1e-8)
    q = np.round((weights - w_min) / scale)
    q = np.clip(q, 0, levels)
    return q * scale + w_min

def gptq_quantize(weights, h_inv, bits=4):
    W_q = weights.copy()
    d_out, d_in = W_q.shape
    levels = (1 << bits) - 1
    w_min = W_q.min(axis=0, keepdims=True)
    w_max = W_q.max(axis=0, keepdims=True)
    scale = np.maximum((w_max - w_min) / levels, 1e-8)
    Hinv = h_inv.copy()
    for i in range(d_in):
        w = W_q[:, i].copy()
        w_quant = np.round((w - w_min[0, i]) / scale[0, i])
        w_quant = np.clip(w_quant, 0, levels) * scale[0, i] + w_min[0, i]
        err = (w - w_quant) / Hinv[i, i]
        W_q[:, i] = w_quant
        if i + 1 < d_in:
            W_q[:, i+1:] -= np.outer(err, Hinv[i, i+1:])
    return W_q
