import numpy as np


def quantize_weights(W, H, bits=4):
    max_val = float(2**(bits - 1) - 1)
    scale = np.max(np.abs(W), axis=0, keepdims=True) / max_val
    scale = np.maximum(scale, 1e-5)
    W_q = np.round(W / scale) * scale
    return W_q, scale
