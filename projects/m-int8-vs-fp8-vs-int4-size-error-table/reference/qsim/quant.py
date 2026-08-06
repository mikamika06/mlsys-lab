import numpy as np
from .fp8 import quantize_to_fp8_vals

def get_scales(x, q_max, axis=1):
    m = np.max(np.abs(x), axis=axis, keepdims=True)
    m = np.maximum(m, 1e-9)
    return m / q_max

def quantize_int8(x, axis=1):
    scales = get_scales(x, 127.0, axis)
    q = np.clip(np.round(x / scales), -127.0, 127.0)
    return q * scales

def quantize_int4(x, axis=1):
    scales = get_scales(x, 7.0, axis)
    q = np.clip(np.round(x / scales), -7.0, 7.0)
    return q * scales

def quantize_fp8(x, axis=1):
    scales = get_scales(x, 448.0, axis)
    q = quantize_to_fp8_vals(x / scales)
    return q * scales
