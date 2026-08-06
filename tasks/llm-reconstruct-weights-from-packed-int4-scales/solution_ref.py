import numpy as np


def dequantize_int4(packed, scales, zero_points, shape, group_size):
    packed = np.asarray(packed, dtype=np.uint8)
    scales = np.asarray(scales, dtype=np.float64)
    zero_points = np.asarray(zero_points, dtype=np.int64)

    n = 1
    for dim in shape:
        n *= dim

    values = np.empty(n, dtype=np.int64)
    for i in range(n):
        byte = int(packed[i // 2])
        if i % 2 == 0:
            values[i] = byte & 15
        else:
            values[i] = (byte >> 4) & 15

    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        group = i // group_size
        out[i] = (values[i] - zero_points[group]) * scales[group]

    return out.reshape(shape)
