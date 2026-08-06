import math
import numpy as np


def int8_linear_forward(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Dynamic int8 Linear forward: per-channel symmetric int8 weight quant
    (fixed ahead of time) + per-tensor dynamic asymmetric uint8 activation
    quant (from this call's own min/max), integer matmul with zero-point
    correction, then dequantize.
    """
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)

    out_dim, in_dim = W.shape
    batch_size = X.shape[0]

    scale_w = []
    for i in range(out_dim):
        max_abs = 0.0
        for j in range(in_dim):
            val = W[i, j]
            if val < 0.0:
                val = -val
            if val > max_abs:
                max_abs = val
        if max_abs == 0.0:
            max_abs = 1.0
        scale_w.append(max_abs / 127.0)

    W_q = np.empty((out_dim, in_dim), dtype=np.int32)
    for i in range(out_dim):
        sw = scale_w[i]
        for j in range(in_dim):
            val = W[i, j] / sw
            if val >= 0.0:
                r = int(math.floor(val + 0.5))
            else:
                r = int(math.ceil(val - 0.5))
            if r < -127:
                r = -127
            elif r > 127:
                r = 127
            W_q[i, j] = r

    x_min = X[0, 0]
    x_max = X[0, 0]
    for i in range(batch_size):
        for j in range(in_dim):
            val = X[i, j]
            if val < x_min:
                x_min = val
            if val > x_max:
                x_max = val

    x_min = float(x_min)
    x_max = float(x_max)
    if x_max == x_min:
        x_max = x_min + 1e-8
    scale_x = (x_max - x_min) / 255.0

    zp_val = -x_min / scale_x
    if zp_val >= 0.0:
        zp = int(math.floor(zp_val + 0.5))
    else:
        zp = int(math.ceil(zp_val - 0.5))
    if zp < 0:
        zp = 0
    elif zp > 255:
        zp = 255
    zero_point = zp

    X_q = np.empty((batch_size, in_dim), dtype=np.int32)
    for i in range(batch_size):
        for j in range(in_dim):
            val = X[i, j] / scale_x + zero_point
            if val >= 0.0:
                r = int(math.floor(val + 0.5))
            else:
                r = int(math.ceil(val - 0.5))
            if r < 0:
                r = 0
            elif r > 255:
                r = 255
            X_q[i, j] = r

    acc = np.empty((batch_size, out_dim), dtype=np.float64)
    for i in range(batch_size):
        for j in range(out_dim):
            s = 0
            for k in range(in_dim):
                s += (int(X_q[i, k]) - zero_point) * int(W_q[j, k])
            acc[i, j] = float(s)

    Y = np.empty((batch_size, out_dim), dtype=np.float64)
    for i in range(batch_size):
        for j in range(out_dim):
            Y[i, j] = acc[i, j] * scale_x * scale_w[j]

    return Y
