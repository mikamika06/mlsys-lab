import numpy as np

MANT_BITS = 23
BIAS = 127


def decompose(x: np.ndarray):
    """BUGGY. Split finite float32 values into (sign, unbiased exponent, significand).

    Two defects hide in here. Find them and fix them; do not rewrite the file
    from scratch, the bit extraction itself is fine.
    """
    b = np.asarray(x, dtype=np.float32).view(np.uint32)
    sign = (b >> 31).astype(np.int64)
    raw_exp = ((b >> MANT_BITS) & np.uint32(0xFF)).astype(np.int64)
    mant = (b & np.uint32((1 << MANT_BITS) - 1)).astype(np.int64)

    exponent = raw_exp + BIAS
    significand = 1.0 + mant.astype(np.float64) / float(1 << MANT_BITS)
    return sign, exponent, significand


def recompose(sign, exponent, significand) -> np.ndarray:
    """Rebuild the float32 vector from the triple."""
    v = (-1.0) ** np.asarray(sign, dtype=np.float64)
    v = v * np.asarray(significand, dtype=np.float64) * np.exp2(
        np.asarray(exponent, dtype=np.float64))
    return v.astype(np.float32)
