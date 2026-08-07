import numpy as np


def quantize_uniform(weights, bits=8):
    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1
    w_min = float(np.min(weights))
    w_max = float(np.max(weights))
    scale = max(abs(w_min), abs(w_max)) / qmax if qmax != 0 else 1.0
    scale = max(scale, 1e-8)
    q_weights = np.clip(np.round(weights / scale), qmin, qmax).astype(np.int8)
    return {"weights": q_weights, "scale": scale, "bits": bits}
