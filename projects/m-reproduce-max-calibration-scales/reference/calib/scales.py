import numpy as np


def max_calibration_scale(tensor, num_bins=2048, quant_max=127.0):
    arr = np.asarray(tensor, dtype=np.float32)
    amax = float(np.max(np.abs(arr)))
    if amax == 0.0:
        return 1.0
    return amax / quant_max
