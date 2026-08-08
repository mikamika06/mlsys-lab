import numpy as np


def enumerate_e2m1_values():
    vals = []
    for bits in range(16):
        sign = (bits >> 3) & 1
        exp = (bits >> 1) & 3
        mant = bits & 1
        if exp == 0:
            val = ((-1.0) ** sign) * (mant / 2.0) * (2.0 ** (-2 + 1))
        else:
            val = ((-1.0) ** sign) * (1.0 + mant / 2.0) * (2.0 ** (exp - 2))
        vals.append(val)
    return np.array(vals, dtype=np.float32)
