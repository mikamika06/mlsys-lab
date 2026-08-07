import math


def _e4m3_scalar(v):
    sign = -1.0 if v < 0 else 1.0
    a = abs(float(v))
    if a == 0:
        return 0.0
    if a >= 448:
        return sign * 448.0
    if a < 2**-6:
        step = 2**-9
        return sign * (round(a / step) * step)
    e = int(math.floor(math.log2(a)))
    step = 2.0 ** (e - 3)
    m = round(a / step)
    if m >= 16:
        e += 1
        m = 8
        step = 2.0 ** (e - 3)
    return sign * min(448.0, m * step)


def quantize_fp8_e4m3_amax(x: list[float]) -> tuple[float, list[float]]:
    amax = 0.0
    for val_in in x:
        val = abs(float(val_in))
        if val > amax:
            amax = val
    if amax == 0:
        return 1.0, [0.0 for _ in x]
    scale = amax / 448.0
    out = []
    for val_in in x:
        v = float(val_in) / scale
        q = _e4m3_scalar(v)
        out.append(q * scale)
    return scale, out
