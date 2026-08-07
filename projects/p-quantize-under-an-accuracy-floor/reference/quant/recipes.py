import numpy as np


def quantize_int8(weights):
    """Performs uniform INT8 symmetric quantization."""
    w_max = np.max(np.abs(weights))
    scale = w_max / 127.0 if w_max > 0 else 1.0
    q = np.clip(np.round(weights / scale), -128, 127).astype(np.int8)
    deq = q.astype(np.float32) * scale
    return q, float(scale), deq


def quantize_int4(weights, scales=None):
    """Performs uniform INT4 symmetric quantization."""
    if scales is None:
        w_max = np.max(np.abs(weights))
        scale = w_max / 7.0 if w_max > 0 else 1.0
    else:
        scale = float(scales)
    q = np.clip(np.round(weights / scale), -8, 7).astype(np.int8)
    deq = q.astype(np.float32) * scale
    return q, float(scale), deq
