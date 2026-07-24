import numpy as np


def _e4m3_scalar(v):
    sign = -1.0 if v < 0 else 1.0
    a = abs(float(v))
    if a == 0:
        return 0.0
    if a >= 448:
        return sign * 448.0
    if a < 2 ** -6:
        step = 2 ** -9
        return sign * (round(a / step) * step)
    e = int(np.floor(np.log2(a)))
    step = 2.0 ** (e - 3)
    m = round(a / step)
    if m >= 16:
        e += 1
        m = 8
        step = 2.0 ** (e - 3)
    return sign * min(448.0, m * step)


def quantize_fp8_e4m3_amax(x):
    x = np.asarray(x, dtype=np.float64)
    amax = float(np.max(np.abs(x)))
    if amax == 0:
        return 1.0, np.zeros_like(x, dtype=np.float64)
    scale = amax / 448.0
    q = np.vectorize(_e4m3_scalar, otypes=[np.float64])(x / scale)
    return scale, q * scale
