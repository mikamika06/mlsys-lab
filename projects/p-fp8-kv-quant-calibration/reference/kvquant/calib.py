import numpy as np

def calibrate_scale(tensors: list[np.ndarray]) -> float:
    max_val = 0.0
    for t in tensors:
        m = np.max(np.abs(t))
        if m > max_val:
            max_val = float(m)
    return max(max_val, 1e-5)
