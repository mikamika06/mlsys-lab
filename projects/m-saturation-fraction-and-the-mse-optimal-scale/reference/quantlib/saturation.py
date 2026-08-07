import numpy as np


def compute_mse_optimal_scale(x, max_val=448.0, num_steps=30):
    """Compute MSE-optimal scale."""
    x_arr = np.asarray(x, dtype=np.float32)
    abs_x = np.abs(x_arr)
    max_abs = np.max(abs_x)
    if max_abs == 0:
        return 1.0
    base_scale = max_abs / max_val
    if base_scale == 0:
        return 1.0
    scales = base_scale * np.linspace(0.5, 1.5, num_steps)
    best_scale = scales[0]
    best_mse = float("inf")
    for s in scales:
        clipped = np.clip(x_arr, -s * max_val, s * max_val)
        mse = np.mean((x_arr - clipped) ** 2)
        if mse < best_mse:
            best_mse = mse
            best_scale = s
    return float(best_scale)


def saturation_fraction(x, scale, max_val=448.0):
    """Compute saturation fraction."""
    x_arr = np.asarray(x, dtype=np.float32)
    if scale <= 0:
        return 1.0
    limit = scale * max_val
    saturated = np.abs(x_arr) > limit
    return float(np.mean(saturated))
