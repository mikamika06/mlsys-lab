import numpy as np


def encode_e4m3(x: np.ndarray) -> np.ndarray:
    """Encodes float32 array into FP8 E4M3 uint8 bit patterns with subnormals."""
    x = np.asarray(x, dtype=np.float32)
    shape = x.shape
    x_flat = x.ravel()

    out = np.zeros(x_flat.shape, dtype=np.uint8)

    max_e4m3 = 448.0
    min_subnormal = 2.0 ** -9

    e4m3_lut = np.zeros(256, dtype=np.float32)
    for byte_val in range(256):
        sign = -1.0 if (byte_val & 0x80) else 1.0
        exp = (byte_val >> 3) & 0x0F
        mant = byte_val & 0x07
        if exp == 0:
            val = sign * (2.0 ** -6) * (mant / 8.0)
        elif exp == 15 and mant == 7:
            val = np.nan
        else:
            val = sign * (2.0 ** (exp - 7)) * (1.0 + mant / 8.0)
        e4m3_lut[byte_val] = val

    valid_indices = np.where(~np.isnan(e4m3_lut))[0]
    valid_vals = e4m3_lut[valid_indices]

    for i, val in enumerate(x_flat):
        if np.isnan(val):
            out[i] = 0x7F
            continue
        sign_bit = 0x80 if np.copysign(1.0, val) < 0 else 0x00
        abs_val = abs(val)

        if abs_val > max_e4m3:
            out[i] = sign_bit | 0x7E
            continue
        if abs_val < min_subnormal / 2.0:
            out[i] = sign_bit | 0x00
            continue

        signed_valid = valid_vals if sign_bit == 0 else -valid_vals
        diffs = np.abs(val - signed_valid)
        best_idx = np.argmin(diffs)
        out[i] = valid_indices[best_idx] if sign_bit == 0 else (valid_indices[best_idx] ^ 0x80)

    return out.reshape(shape)


def decode_e4m3(u: np.ndarray) -> np.ndarray:
    """Decodes FP8 E4M3 uint8 bit patterns to float32 values."""
    u = np.asarray(u, dtype=np.uint8)
    e4m3_lut = np.zeros(256, dtype=np.float32)
    for byte_val in range(256):
        sign = -1.0 if (byte_val & 0x80) else 1.0
        exp = (byte_val >> 3) & 0x0F
        mant = byte_val & 0x07
        if exp == 0:
            val = sign * (2.0 ** -6) * (mant / 8.0)
        elif exp == 15 and mant == 7:
            val = np.nan
        else:
            val = sign * (2.0 ** (exp - 7)) * (1.0 + mant / 8.0)
        e4m3_lut[byte_val] = val

    return e4m3_lut[u]
