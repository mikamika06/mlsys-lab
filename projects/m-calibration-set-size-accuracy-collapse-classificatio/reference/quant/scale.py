import numpy as np


def reconstruct_scales(tensor_stats, num_bits=8):
    scales = []
    qmin = -(2 ** (num_bits - 1))
    qmax = (2 ** (num_bits - 1)) - 1
    for stat in tensor_stats:
        min_val = float(stat["min"])
        max_val = float(stat["max"])
        abs_max = max(abs(min_val), abs(max_val))
        if abs_max == 0.0:
            scale = 1.0
        else:
            scale = abs_max / qmax
        scales.append(float(scale))
    return scales
