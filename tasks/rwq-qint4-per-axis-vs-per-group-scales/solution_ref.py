import numpy as np


def _sym_int4_dequant(x: np.ndarray, scale: np.ndarray) -> np.ndarray:
    shape = x.shape
    if len(shape) == 2:
        rows, cols = shape
        out_arr = np.zeros((rows, cols), dtype=x.dtype)
        for i in range(rows):
            s = scale[i, 0]
            s_safe = 1.0 if s == 0 else s
            for j in range(cols):
                val = x[i, j] / s_safe
                r = round(val)
                if r < -7:
                    r = -7
                elif r > 7:
                    r = 7
                out_arr[i, j] = r * s_safe
        return out_arr
    elif len(shape) == 3:
        d0, d1, d2 = shape
        out_arr = np.zeros((d0, d1, d2), dtype=x.dtype)
        for i in range(d0):
            for j in range(d1):
                s = scale[i, j, 0]
                s_safe = 1.0 if s == 0 else s
                for k in range(d2):
                    val = x[i, j, k] / s_safe
                    r = round(val)
                    if r < -7:
                        r = -7
                    elif r > 7:
                        r = 7
                    out_arr[i, j, k] = r * s_safe
        return out_arr
    raise ValueError("Unsupported shape")


def qint4_granularity_mse(W: np.ndarray, group_size: int = 32):
    W = np.asarray(W, dtype=np.float64)
    rows, cols = W.shape

    amax_axis = np.zeros(rows, dtype=W.dtype)
    for i in range(rows):
        m = 0.0
        for j in range(cols):
            v = W[i, j]
            if v < 0:
                v = -v
            if j == 0 or v > m:
                m = v
        amax_axis[i] = m

    scale_axis = np.zeros(rows, dtype=W.dtype)
    for i in range(rows):
        scale_axis[i] = amax_axis[i] / 7.0

    scale_axis_2d = np.zeros((rows, 1), dtype=W.dtype)
    for i in range(rows):
        scale_axis_2d[i, 0] = scale_axis[i]

    deq_axis = _sym_int4_dequant(W, scale_axis_2d)

    sum_sq_axis = 0.0
    for i in range(rows):
        for j in range(cols):
            diff = W[i, j] - deq_axis[i, j]
            sum_sq_axis += diff * diff
    mse_per_axis = float(sum_sq_axis / (rows * cols))

    ng = cols // group_size
    Wg = W.reshape(rows, ng, group_size)

    amax_group = np.zeros((rows, ng), dtype=W.dtype)
    for i in range(rows):
        for g in range(ng):
            m = 0.0
            for k in range(group_size):
                v = Wg[i, g, k]
                if v < 0:
                    v = -v
                if k == 0 or v > m:
                    m = v
            amax_group[i, g] = m

    scale_group = np.zeros((rows, ng), dtype=W.dtype)
    for i in range(rows):
        for g in range(ng):
            scale_group[i, g] = amax_group[i, g] / 7.0

    scale_group_3d = np.zeros((rows, ng, 1), dtype=W.dtype)
    for i in range(rows):
        for g in range(ng):
            scale_group_3d[i, g, 0] = scale_group[i, g]

    deq_group = _sym_int4_dequant(Wg, scale_group_3d).reshape(rows, cols)

    sum_sq_group = 0.0
    for i in range(rows):
        for j in range(cols):
            diff = W[i, j] - deq_group[i, j]
            sum_sq_group += diff * diff
    mse_per_group = float(sum_sq_group / (rows * cols))

    return mse_per_axis, mse_per_group
