import numpy as np


def optimize_zero_point(W, scale, bits, iters):
    W = np.asarray(W, dtype=np.float64)
    qmax = (1 << bits) - 1
    z = 0

    def reconstruct(z_value):
        q = np.clip(np.rint(W / scale) + z_value, 0, qmax)
        return scale * (q - z_value)

    for _ in range(iters):
        best_z = z
        best_err = None
        for candidate in range(z - 2, z + 3):
            err = np.sum((W - reconstruct(candidate)) ** 2)
            if best_err is None or err < best_err:
                best_err = err
                best_z = candidate
        z = best_z

    return reconstruct(z).astype(np.float64), z
