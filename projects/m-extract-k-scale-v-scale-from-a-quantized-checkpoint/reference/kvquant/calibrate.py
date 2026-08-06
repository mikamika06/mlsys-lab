import numpy as np

def absmax_calibrate(activations):
    arr = np.array(activations, dtype=np.float32)
    max_val = np.max(np.abs(arr))
    if max_val == 0.0:
        return 1.0
    return float(max_val / 127.0)
