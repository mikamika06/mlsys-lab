import numpy as np


def dequantize_int4(packed, scales, zero_points, shape, group_size):
    packed = np.asarray(packed, dtype=np.uint8)
    scales = np.asarray(scales, dtype=np.float64)
    zero_points = np.asarray(zero_points, dtype=np.int64)

    n = int(np.prod(shape))
    values = np.empty(n, dtype=np.int64)

    values[0::2] = packed & 15
    values[1::2] = (packed >> 4) & 15

    groups = np.arange(n) // group_size
    out = (values - zero_points[groups]) * scales[groups]
    return out.astype(np.float64).reshape(shape)
