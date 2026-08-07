import numpy as np


def compute_mse_scale(tensor, max_val=448.0):
    arr = np.abs(np.array(tensor, dtype=np.float32))
    if arr.size == 0:
        return 1.0
    abs_max = np.max(arr)
    if abs_max == 0:
        return 1.0
    candidates = np.linspace(abs_max * 0.1, abs_max * 1.5, num=20)
    best_scale = candidates[0]
    best_mse = float("inf")
    for c in candidates:
        scale = max_val / c
        clipped = np.clip(arr * scale, 0, max_val)
        dequant = clipped / scale
        mse = np.mean((arr - dequant) ** 2)
        if mse < best_mse:
            best_mse = mse
            best_scale = c
    return float(best_scale)
