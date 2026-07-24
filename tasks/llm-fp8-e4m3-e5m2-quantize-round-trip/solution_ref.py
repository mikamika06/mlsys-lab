import numpy as np


def _convert(v, fmt):
    if fmt == "e4m3":
        eb, mb, bias = 4, 3, 7
    else:
        eb, mb, bias = 5, 2, 15

    bits = np.frombuffer(np.float32(v).tobytes(), dtype=np.uint32)[0]
    sign = int(bits >> 31)
    exp = int((bits >> 23) & 255)
    frac = int(bits & 0x7fffff)

    if exp == 0:
        return 0.0

    e = exp - 127 + bias
    max_e = (1 << eb) - 2

    if e > max_e:
        e = max_e
        m = (1 << mb) - 1
    elif e <= 0:
        e = 0
        m = 0
    else:
        shift = 23 - mb
        m = frac >> shift
        rem = frac & ((1 << shift) - 1)
        half = 1 << (shift - 1)
        if rem > half or (rem == half and (m & 1)):
            m += 1
            if m == (1 << mb):
                m = 0
                e += 1

    if e > max_e:
        e = max_e
        m = (1 << mb) - 1

    sign_value = -1.0 if sign else 1.0
    if e == 0:
        return 0.0
    return sign_value * (2.0 ** (e - bias)) * (1.0 + m / (2.0 ** mb))


def fp8_roundtrip(x: np.ndarray, fmt: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    out = np.empty(x.shape, dtype=np.float64)
    for idx, value in np.ndenumerate(x):
        out[idx] = _convert(value, fmt)
    return out
