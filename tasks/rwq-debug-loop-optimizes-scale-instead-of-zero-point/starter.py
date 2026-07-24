import numpy as np


def optimize_zero_point(W, scale, bits, iters):
    # TODO: broken debug fix. This loop incorrectly changes scale and keeps z fixed.
    W = np.asarray(W, dtype=np.float64)
    qmax = (1 << bits) - 1
    z = 0
    s = float(scale)

    for _ in range(iters):
        q = np.clip(np.rint(W / s) + z, 0, qmax)
        recon = s * (q - z)
        grad = np.mean((recon - W) * (q - z))
        s = s - 0.1 * grad

    q = np.clip(np.rint(W / s) + z, 0, qmax)
    return (s * (q - z)).astype(np.float64), z
