import math
import numpy as np


def optimize_zero_point(W, scale, bits, iters):
    W = np.asarray(W, dtype=np.float64)
    qmax = (1 << bits) - 1
    z = 0

    shape = W.shape
    flat_W = W.ravel()
    n = flat_W.size

    def reconstruct(z_value):
        q = np.empty(n, dtype=np.float64)
        rec = np.empty(n, dtype=np.float64)
        for i in range(n):
            val = round(flat_W[i] / scale) + z_value
            if val < 0:
                val = 0
            elif val > qmax:
                val = qmax
            q[i] = val
            rec[i] = scale * (q[i] - z_value)
        return rec.reshape(shape)

    def reconstruct_flat(z_value):
        rec = np.empty(n, dtype=np.float64)
        for i in range(n):
            val = round(flat_W[i] / scale) + z_value
            if val < 0:
                val = 0
            elif val > qmax:
                val = qmax
            rec[i] = scale * (val - z_value)
        return rec

    for _ in range(iters):
        best_z = z
        best_err = None
        for candidate in range(z - 2, z + 3):
            rec_candidate = reconstruct_flat(candidate)
            err = 0.0
            for i in range(n):
                diff = flat_W[i] - rec_candidate[i]
                err += diff * diff
            if best_err is None or err < best_err:
                best_err = err
                best_z = candidate
        z = best_z

    return reconstruct(z).astype(np.float64), z
