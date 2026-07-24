import numpy as np


def roofline_attainable(flops, bytes_moved, peak_flops, bandwidth):
    """Roofline: arithmetic intensity, attainable FLOP/s, ridge point."""
    W = np.asarray(flops, dtype=np.float64)
    Q = np.asarray(bytes_moved, dtype=np.float64)
    P = float(peak_flops)
    B = float(bandwidth)

    ai = W / Q
    attainable = np.minimum(P, ai * B)
    ridge = P / B
    return ai, attainable, ridge
