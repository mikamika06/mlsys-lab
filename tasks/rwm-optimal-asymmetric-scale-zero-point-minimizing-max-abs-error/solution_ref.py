import numpy as np


def derive_affine_qparams(x: np.ndarray, nbits: int) -> tuple:
    """
    Standard asymmetric min-max affine quantization params, with the group's
    range extended to always include zero so the mapping is lossless at 0
    and the round-to-nearest reconstruction error is bounded by scale/2.
    """
    x = np.asarray(x, dtype=np.float64)
    qmax = (1 << nbits) - 1
    mn = min(0.0, float(np.min(x)))
    mx = max(0.0, float(np.max(x)))
    if mx == mn:
        scale = 1.0
    else:
        scale = (mx - mn) / qmax
    zero_point = int(np.clip(round(-mn / scale), 0, qmax))
    return float(scale), zero_point
