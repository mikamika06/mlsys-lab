import numpy as np


def encode_e4m3(arr: np.ndarray) -> np.ndarray:
    flat = np.asarray(arr, dtype=np.float32).ravel()
    out = np.zeros(flat.shape, dtype=np.uint8)
    max_val = 448.0
    min_subnormal = 2.0 ** (-9)

    for i, x in enumerate(flat):
        if np.isnan(x):
            out[i] = 0x7F
            continue

        sign = 0x80 if np.copysign(1.0, x) < 0 else 0x00
        abs_x = abs(float(x))

        if abs_x >= max_val:
            out[i] = sign | 0x7E
            continue

        if abs_x < min_subnormal / 2.0:
            out[i] = sign
            continue

        if abs_x < 2.0 ** (-6):
            frac = abs_x / (2.0 ** (-9))
            mant = int(round(frac))
            if mant > 7:
                out[i] = sign | 0x08
            else:
                out[i] = sign | (mant & 0x07)
            continue

        exp = int(np.floor(np.log2(abs_x)))
        biased_exp = exp + 7
        mant = int(round(((abs_x / (2.0 ** exp)) - 1.0) * 8.0))

        if mant == 8:
            mant = 0
            biased_exp += 1

        if biased_exp >= 15 and mant > 6:
            biased_exp = 15
            mant = 6

        out[i] = sign | ((biased_exp & 0x0F) << 3) | (mant & 0x07)

    return out.reshape(arr.shape)


def decode_e4m3(bytes_arr: np.ndarray) -> np.ndarray:
    u = np.asarray(bytes_arr, dtype=np.uint8).ravel()
    out = np.zeros(u.shape, dtype=np.float32)

    for i, b in enumerate(u):
        sign = -1.0 if (b & 0x80) else 1.0
        exp = (b >> 3) & 0x0F
        mant = b & 0x07

        if exp == 15 and mant == 7:
            out[i] = np.nan
        elif exp == 0:
            out[i] = sign * (2.0 ** (-6)) * (mant / 8.0)
        else:
            out[i] = sign * (2.0 ** (exp - 7)) * (1.0 + mant / 8.0)

    return out.reshape(bytes_arr.shape)
