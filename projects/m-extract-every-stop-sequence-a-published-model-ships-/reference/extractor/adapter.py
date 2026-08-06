import numpy as np


def apply_adapter_and_forward(base_weights, adapter_weights, x, alpha=1.0):
    w_base = np.array(base_weights, dtype=float)
    a, b = adapter_weights
    w_adapted = w_base + alpha * (np.array(a, dtype=float) @ np.array(b, dtype=float))
    x_arr = np.array(x, dtype=float)
    return x_arr @ w_adapted
