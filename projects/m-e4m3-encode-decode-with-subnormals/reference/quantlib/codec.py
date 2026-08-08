import numpy as np


def encode_e4m3(x):
    x = np.asarray(x, dtype=np.float32)
    signs = (x < 0).astype(np.uint8)
    abs_x = np.abs(x)
    out = np.zeros_like(x, dtype=np.uint8)
    for i in range(x.size):
        val = abs_x.flat[i]
        if val == 0:
            code = 0
        else:
            e = int(np.floor(np.log2(val))) + 7
            if e < 0:
                e = 0
                m = int(round(val * (2.0 ** 6) * 8.0))
                m = min(7, max(0, m))
            elif e > 15:
                e = 15
                m = 7
            else:
                m = int(round((val / (2.0 ** (e - 7)) - 1.0) * 8.0))
                if m > 7:
                    m = 7
                    e += 1
                    if e > 15:
                        e = 15
                        m = 7
                m = max(0, m)
            code = (e << 3) | m
        out.flat[i] = (signs.flat[i] << 7) | code
    return out


def decode_e4m3(b):
    b = np.asarray(b, dtype=np.uint8)
    out = np.zeros(b.shape, dtype=np.float32)
    for i in range(b.size):
        byte = b.flat[i]
        s = (byte >> 7) & 1
        e = (byte >> 3) & 15
        m = byte & 7
        if e == 0:
            val = (m / 8.0) * (2.0 ** -6)
        else:
            val = (1.0 + m / 8.0) * (2.0 ** (e - 7))
        if s:
            val = -val
        out.flat[i] = val
    return out
