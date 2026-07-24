import numpy as np

MANT_BITS = 23
BIAS = 127


def decompose(x: np.ndarray):
    """Split finite float32 values into (sign, unbiased exponent, significand)."""
    b = np.asarray(x, dtype=np.float32).view(np.uint32)
    sign = (b >> 31).astype(np.int64)
    raw_exp = ((b >> MANT_BITS) & np.uint32(0xFF)).astype(np.int64)
    mant = (b & np.uint32((1 << MANT_BITS) - 1)).astype(np.int64)

    is_sub = raw_exp == 0
    # FIX 1: the bias is SUBTRACTED, and subnormals share the exponent 1 - BIAS.
    exponent = np.where(is_sub, np.int64(1 - BIAS), raw_exp - BIAS)
    # FIX 2: the leading bit is implicit 1 only for normals; subnormals lead with 0.
    lead = np.where(is_sub, 0.0, 1.0)
    significand = lead + mant.astype(np.float64) / float(1 << MANT_BITS)
    return sign, exponent, significand


def recompose(sign, exponent, significand) -> np.ndarray:
    """Rebuild the float32 vector from the triple."""
    v = (-1.0) ** np.asarray(sign, dtype=np.float64)
    v = v * np.asarray(significand, dtype=np.float64) * np.exp2(
        np.asarray(exponent, dtype=np.float64))
    return v.astype(np.float32)
