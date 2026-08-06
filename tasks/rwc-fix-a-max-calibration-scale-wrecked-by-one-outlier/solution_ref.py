import math
import numpy as np


def calibrate_scale_and_dequantize(x, qmax=127, percentile=99.0):
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    
    abs_x = np.empty(n, dtype=np.float64)
    for i in range(n):
        val = x[i]
        if val < 0.0:
            abs_x[i] = -val
        else:
            abs_x[i] = val

    sorted_abs = np.empty(n, dtype=np.float64)
    for i in range(n):
        sorted_abs[i] = abs_x[i]
    
    for i in range(1, n):
        key = sorted_abs[i]
        j = i - 1
        while j >= 0 and sorted_abs[j] > key:
            sorted_abs[j + 1] = sorted_abs[j]
            j -= 1
        sorted_abs[j + 1] = key

    rank = (percentile / 100.0) * (n - 1)
    lower_idx = int(math.floor(rank))
    upper_idx = int(math.ceil(rank))
    weight = rank - lower_idx

    if lower_idx == upper_idx:
        amax = float(sorted_abs[lower_idx])
    else:
        amax = float((1.0 - weight) * sorted_abs[lower_idx] + weight * sorted_abs[upper_idx])

    scale = amax / qmax

    q = np.empty(n, dtype=np.float64)
    reconstructed = np.empty(n, dtype=np.float64)
    for i in range(n):
        val = x[i] / scale
        if val >= 0.0:
            rounded = math.floor(val + 0.5)
        else:
            rounded = math.ceil(val - 0.5)
        
        if rounded < -qmax:
            clipped = -qmax
        elif rounded > qmax:
            clipped = qmax
        else:
            clipped = rounded
            
        q[i] = clipped
        reconstructed[i] = q[i] * scale

    return amax, scale, reconstructed
