import math
import numpy as np

def decode_e4m3(codes):
    """Decode an array of E4M3 uint8 bit patterns to float64 values."""
    codes = np.asarray(codes, dtype=np.uint8)
    n = len(codes)
    result = np.zeros(n, dtype=np.float64)

    for i in range(n):
        code = codes[i]
        sign_bit = code >> 7
        exp_bits = (code >> 3) & 0x0F
        man_bits = float(code & 0x07)

        sign = 1.0 if sign_bit == 0 else -1.0

        if exp_bits != 0:
            val = 1.0 + man_bits / 8.0
            p = int(exp_bits) - 7
            result[i] = sign * math.ldexp(val, p)
        elif man_bits != 0.0:
            val = man_bits / 8.0
            result[i] = sign * math.ldexp(val, -6)
        else:
            result[i] = 0.0 if sign_bit == 0 else -0.0

    return result
