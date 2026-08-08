import numpy as np


def derive_symmetric(weights, bits=4):
    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1
    max_val = np.max(np.abs(weights))
    scale = max_val / qmax if max_val != 0 else 1.0
    zero_point = 0
    return float(scale), int(zero_point)


def derive_affine(weights, bits=4):
    qmin = 0
    qmax = (1 << bits) - 1
    min_val = float(np.min(weights))
    max_val = float(np.max(weights))
    if min_val == max_val:
        scale = 1.0
        zero_point = qmin
    else:
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = int(np.round(qmin - min_val / scale))
        zero_point = int(np.clip(zero_point, qmin, qmax))
    return float(scale), int(zero_point)
