import numpy as np


def fp16_roundtrip(x):
    return np.asarray(x, dtype=np.float32).astype(np.float16).astype(np.float32)


def bf16_roundtrip(x):
    bits = np.asarray(x, dtype=np.float32).view(np.uint32)
    return (((bits >> 16) << 16).view(np.float32))


def _fp8_table():
    values = []
    for code in range(256):
        sign = -1.0 if code & 0x80 else 1.0
        exp = (code >> 3) & 0x0F
        mant = code & 0x07
        if exp == 0:
            value = sign * (mant / 8.0) * (2.0 ** -6)
        elif exp == 15:
            value = sign * (240.0 + mant * 16.0)
        else:
            value = sign * (1.0 + mant / 8.0) * (2.0 ** (exp - 7))
        values.append(np.float32(value))
    return values


def fp8_e4m3_roundtrip(x):
    table = _fp8_table()
    arr = np.asarray(x, dtype=np.float32)
    result = []
    for value in arr.ravel():
        best = 0
        best_err = float("inf")
        for code, candidate in enumerate(table):
            err = abs(float(value) - float(candidate))
            if err < best_err:
                best_err = err
                best = code
        result.append(table[best])
    return np.asarray(result, dtype=np.float32).reshape(arr.shape)
