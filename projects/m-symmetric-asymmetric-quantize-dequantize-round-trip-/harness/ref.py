import numpy as np

np.random.seed(42)
SKEWED_DATA = np.abs(np.random.randn(1000) * 10)
SKEWED_DATA[::10] = 0.0

WEIGHTS = np.random.randn(16, 8, 3, 3)
WEIGHTS[0] *= 100.0
WEIGHTS[1] *= 0.01

def calc_scale_zp_asymmetric(min_val, max_val, bits=8):
    min_val = min(0.0, float(min_val))
    max_val = max(0.0, float(max_val))
    q_max = (1 << bits) - 1
    if max_val == min_val:
        return 1.0, 0
    scale = (max_val - min_val) / q_max
    zp = int(np.round(-min_val / scale))
    return scale, zp

def calc_scale_symmetric(max_abs_val, bits=8):
    q_max = (1 << (bits - 1)) - 1
    if max_abs_val == 0:
        return 1.0
    return float(max_abs_val) / q_max

def quantize_asymmetric(x, scale, zp, bits=8):
    q_max = (1 << bits) - 1
    q = np.round(x / scale) + zp
    return np.clip(q, 0, q_max).astype(np.int32)

def dequantize_asymmetric(x_q, scale, zp):
    return (x_q.astype(np.float32) - zp) * scale

def quantize_symmetric(x, scale, bits=8):
    q_max = (1 << (bits - 1)) - 1
    q_min = -q_max
    q = np.round(x / scale)
    return np.clip(q, q_min, q_max).astype(np.int32)

def dequantize_symmetric(x_q, scale):
    return x_q.astype(np.float32) * scale

def per_channel_weight_scales(w, bits=8):
    max_abs = np.max(np.abs(w), axis=(1, 2, 3), keepdims=True)
    max_abs = np.maximum(max_abs, 1e-9)
    q_max = (1 << (bits - 1)) - 1
    return max_abs / q_max

def fused_requantize_scale(s_in, s_w, s_out):
    return (s_in * s_w) / s_out
