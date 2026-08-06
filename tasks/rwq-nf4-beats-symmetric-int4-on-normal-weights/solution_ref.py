import numpy as np
import math

_NF4 = np.array([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
], dtype=np.float64)


def nf4_vs_int4_mse(w: np.ndarray):
    """
    Quantize a block of near-Gaussian weights two ways and return their
    reconstruction MSEs as (mse_nf4, mse_int4):

    - NF4: normalize by absmax, snap each value to the nearest of the 16
      fixed NF4 codebook levels, scale back by absmax.
    - Symmetric INT4: scale = absmax / 7, round(w / scale) clipped to
      [-8, 7], dequantize by * scale.
    """
    w = np.asarray(w, dtype=np.float64)
    n = w.shape[0]

    max_val = 0.0
    for i in range(n):
        val = w[i]
        if val < 0.0:
            val = -val
        if val > max_val:
            max_val = val
    absmax = float(max_val) or 1.0

    deq_nf4_list = []
    for i in range(n):
        wn = w[i] / absmax
        best_diff = float("inf")
        best_code = 0.0
        for j in range(16):
            code = _NF4[j]
            diff = wn - code
            if diff < 0.0:
                diff = -diff
            if diff < best_diff:
                best_diff = diff
                best_code = code
        deq_nf4_list.append(best_code * absmax)

    sum_sq_nf4 = 0.0
    for i in range(n):
        diff = w[i] - deq_nf4_list[i]
        sum_sq_nf4 += diff * diff
    mse_nf4 = float(sum_sq_nf4 / n)

    scale = absmax / 7.0
    sum_sq_int4 = 0.0
    for i in range(n):
        val = w[i] / scale
        rounded = round(val)
        if rounded < -8.0:
            rounded = -8.0
        elif rounded > 7.0:
            rounded = 7.0
        deq_val = rounded * scale
        diff = w[i] - deq_val
        sum_sq_int4 += diff * diff
    mse_int4 = float(sum_sq_int4 / n)

    return mse_nf4, mse_int4
