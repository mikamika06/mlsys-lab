import numpy as np

def decode_e4m3(codes):
    """Decode an array of E4M3 uint8 bit patterns to float64 values."""
    codes = np.asarray(codes, dtype=np.uint8)

    sign_bit = (codes >> 7).astype(np.uint8)
    exp_bits = ((codes >> 3) & 0x0F).astype(np.uint16)
    man_bits = (codes & 0x07).astype(np.float64)

    sign = np.where(sign_bit == 0, 1.0, -1.0)

    result = np.zeros(len(codes), dtype=np.float64)

    # Normalized: exponent != 0
    norm = exp_bits != 0
    result[norm] = sign[norm] * np.ldexp(
        1.0 + man_bits[norm] / 8.0,
        (exp_bits[norm].astype(np.int16) - 7).astype(np.int16),
    )

    # Subnormal: exponent == 0, mantissa != 0
    sub = (exp_bits == 0) & (man_bits != 0.0)
    result[sub] = sign[sub] * np.ldexp(man_bits[sub] / 8.0, -6)

    # Zero: exponent == 0, mantissa == 0
    zero = (exp_bits == 0) & (man_bits == 0.0)
    result[zero] = np.where(sign_bit[zero] == 0, 0.0, -0.0)

    return result
