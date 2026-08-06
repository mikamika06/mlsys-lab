import math
import numpy as np


def stable_log_add_exp(a, b):
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    a_bcast, b_bcast = np.broadcast_arrays(a_arr, b_arr)
    out = np.zeros(a_bcast.shape, dtype=np.float64)

    a_flat = a_bcast.flat
    b_flat = b_bcast.flat
    out_flat = out.flat

    for i in range(out.size):
        va = float(a_flat[i])
        vb = float(b_flat[i])
        if va >= vb:
            m = va
            diff = va - vb
        else:
            m = vb
            diff = vb - va
        out_flat[i] = m + math.log1p(math.exp(-diff))

    return out
