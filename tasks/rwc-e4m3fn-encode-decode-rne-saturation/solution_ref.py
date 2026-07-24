import numpy as np


def _decode_scalar(code):
    sign = -1.0 if code & 0x80 else 1.0
    e = (code >> 3) & 0x0F
    m = code & 0x07
    if e == 0:
        return sign * (m / 8.0) * (2.0 ** -6)
    if e == 15 and m == 7:
        return np.nan
    return sign * (1.0 + m / 8.0) * (2.0 ** (e - 7))


def decode_e4m3fn(codes):
    arr = np.asarray(codes, dtype=np.uint8)
    out = np.empty(arr.shape, dtype=np.float32)
    for idx, code in np.ndenumerate(arr):
        out[idx] = _decode_scalar(int(code))
    return out


def _round_code(value):
    sign = 0x80 if value < 0 else 0
    mag = abs(float(value))
    candidates = []
    for code in range(256):
        if (code & 0x80) == sign and (code & 0x7F) != 0x7F:
            candidates.append(code)
    best = candidates[0]
    best_dist = abs(abs(_decode_scalar(best)) - mag)
    for code in candidates[1:]:
        dist = abs(abs(_decode_scalar(code)) - mag)
        if dist < best_dist:
            best = code
            best_dist = dist
        elif dist == best_dist:
            if (code & 7) % 2 == 0 and (best & 7) % 2 == 1:
                best = code
    return best


def encode_e4m3fn(x):
    arr = np.asarray(x)
    out = np.empty(arr.shape, dtype=np.uint8)
    for idx, value in np.ndenumerate(arr):
        out[idx] = _round_code(float(value))
    return out
