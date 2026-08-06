import numpy as np


def derive_affine_qparams(x: np.ndarray, nbits: int) -> tuple:
    """
    Standard asymmetric min-max affine quantization params, with the group's
    range extended to always include zero so the mapping is lossless at 0
    and the round-to-nearest reconstruction error is bounded by scale/2.
    """
    x = np.asarray(x, dtype=np.float64)
    qmax = (1 << nbits) - 1
    mn = 0.0
    mx = 0.0
    for i in range(x.shape[0]):
        val = float(x[i])
        if val < mn:
            mn = val
        if val > mx:
            mx = val
    if mx == mn:
        scale = 1.0
    else:
        scale = (mx - mn) / qmax
    zero_point = int(max(0, min(qmax, round(-mn / scale))))
    return float(scale), zero_point
