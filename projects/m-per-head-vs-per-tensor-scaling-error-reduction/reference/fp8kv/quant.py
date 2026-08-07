import numpy as np


def simulate_e4m3(x, scale):
    q = np.clip(np.round(x * scale), -448.0, 448.0)
    return q / scale


def get_per_tensor_scale(x, max_val=448.0):
    m = np.max(np.abs(x))
    return float(max_val / m) if m > 0 else 1.0


def get_per_head_scale(x, max_val=448.0):
    m = np.max(np.abs(x), axis=(0, 2), keepdims=True)
    return np.where(m > 0, max_val / m, 1.0)
