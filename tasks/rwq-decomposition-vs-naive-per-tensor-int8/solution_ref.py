import numpy as np
import math

def compare_quantization(X, threshold=6.0):
    """Compare LLM.int8() decomposition vs naive per-tensor int8."""
    X = np.asarray(X, dtype=np.float64)
    rows, cols = X.shape

    abs_max = 0.0
    for i in range(rows):
        for j in range(cols):
            val = abs(X[i, j])
            if val > abs_max:
                abs_max = val

    scale_n = abs_max / 127.0 if abs_max > 0 else 1.0

    sq_err_sum_n = 0.0
    for i in range(rows):
        for j in range(cols):
            val = X[i, j]
            scaled = val / scale_n
            rounded = math.floor(scaled + 0.5) if scaled >= 0 else math.ceil(scaled - 0.5)
            if rounded > 127.0:
                clipped = 127
            elif rounded < -128.0:
                clipped = -128
            else:
                clipped = int(rounded)
            Xr_n_ij = float(np.int8(clipped)) * scale_n
            diff = val - Xr_n_ij
            sq_err_sum_n += diff * diff

    mse_naive = float(sq_err_sum_n / (rows * cols))

    outlier = [False] * cols
    has_outlier = False
    has_non_outlier = False

    for j in range(cols):
        c_max = 0.0
        for i in range(rows):
            val = abs(X[i, j])
            if val > c_max:
                c_max = val
        if c_max > threshold:
            outlier[j] = True
            has_outlier = True
        else:
            has_non_outlier = True

    s = 0.0
    if has_non_outlier:
        for i in range(rows):
            for j in range(cols):
                if not outlier[j]:
                    val = abs(X[i, j])
                    if val > s:
                        s = val
        s = s / 127.0 if s > 0 else 1.0

    sq_err_sum_decomp = 0.0
    for i in range(rows):
        for j in range(cols):
            val = X[i, j]
            if outlier[j]:
                Xr_ij = float(np.float64(np.float16(val)))
            else:
                scaled = val / s
                rounded = math.floor(scaled + 0.5) if scaled >= 0 else math.ceil(scaled - 0.5)
                if rounded > 127.0:
                    clipped = 127
                elif rounded < -128.0:
                    clipped = -128
                else:
                    clipped = int(rounded)
                Xr_ij = float(np.int8(clipped)) * s
            diff = val - Xr_ij
            sq_err_sum_decomp += diff * diff

    mse_decomp = float(sq_err_sum_decomp / (rows * cols))

    return (mse_decomp, mse_naive)
