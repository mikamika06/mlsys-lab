import numpy as np


def compute_scale(tensor, qmin, qmax):
    max_val = np.max(np.abs(tensor))
    if max_val == 0.0:
        return np.array(1.0, dtype=np.float32)
    return max_val / float(qmax)


def quantize(tensor, scale, qmin, qmax):
    scaled = np.round(tensor / scale)
    return np.clip(scaled, qmin, qmax).astype(np.int32)


def dequantize(codes, scale):
    return codes.astype(np.float32) * scale
