import math
import numpy as np


def compare_quant_errors(W):
    W = np.asarray(W, dtype=np.float64)
    rows = W.shape[0]
    cols = W.shape[1]

    max_abs_tensor = 0.0
    for i in range(rows):
        for j in range(cols):
            val = W[i, j]
            abs_val = val if val >= 0.0 else -val
            if abs_val > max_abs_tensor:
                max_abs_tensor = abs_val

    tensor_scale = max_abs_tensor / 127.0

    tensor_q = np.empty((rows, cols), dtype=np.float64)
    for i in range(rows):
        for j in range(cols):
            val = W[i, j] / tensor_scale
            r = round(val)
            if r < -127.0:
                r = -127.0
            elif r > 127.0:
                r = 127.0
            tensor_q[i, j] = r

    tensor_out = tensor_q * tensor_scale

    channel_scale = np.empty((rows, 1), dtype=np.float64)
    for i in range(rows):
        max_abs_row = 0.0
        for j in range(cols):
            val = W[i, j]
            abs_val = val if val >= 0.0 else -val
            if abs_val > max_abs_row:
                max_abs_row = abs_val
        channel_scale[i, 0] = max_abs_row / 127.0

    channel_q = np.empty((rows, cols), dtype=np.float64)
    for i in range(rows):
        scale = channel_scale[i, 0]
        for j in range(cols):
            val = W[i, j] / scale
            r = round(val)
            if r < -127.0:
                r = -127.0
            elif r > 127.0:
                r = 127.0
            channel_q[i, j] = r

    channel_out = channel_q * channel_scale

    return tensor_out, channel_out
