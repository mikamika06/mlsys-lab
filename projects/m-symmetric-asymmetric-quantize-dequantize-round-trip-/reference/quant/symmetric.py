import numpy as np


def quantize_symmetric(x, qmin=-128, qmax=127):
    max_val = np.max(np.abs(x))
    scale = max_val / float(max(abs(qmin), qmax))
    if scale == 0.0:
        scale = 1.0
    q = np.clip(np.round(x / scale), qmin, qmax).astype(np.int32)
    return q, scale


def dequantize_symmetric(q, scale):
    return q.astype(np.float32) * scale


def quantize_asymmetric(x, qmin=0, qmax=255):
    xmin = np.min(x)
    xmax = np.max(x)
    if xmin == xmax:
        scale = 1.0
        zero_point = qmin
    else:
        scale = (xmax - xmin) / float(qmax - qmin)
        zero_point = np.clip(np.round(qmin - xmin / scale), qmin, qmax)
        zero_point = int(zero_point)
    q = np.clip(np.round(x / scale + zero_point), qmin, qmax).astype(np.int32)
    return q, scale, zero_point


def dequantize_asymmetric(q, scale, zero_point, qmin=0, qmax=255):
    return (q.astype(np.float32) - zero_point) * scale
