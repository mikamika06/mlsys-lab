import numpy as np

def calibrate_w8a8(tensor):
    abs_max = float(np.max(np.abs(tensor)))
    scale = abs_max / 127.0 if abs_max > 0 else 1.0
    zero_point = 0
    return scale, zero_point
