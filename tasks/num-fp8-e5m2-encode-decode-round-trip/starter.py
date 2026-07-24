import numpy as np


def encode_e5m2(values: np.ndarray) -> np.ndarray:
    """Quantize float32 values to E5M2 fp8 codes.

    Layout: sign(1) | exponent(5, bias 15) | mantissa(2).
    Normal:    v = (-1)^s * 2^(e-15) * (1 + m/4),  1 <= e <= 30
    Subnormal: v = (-1)^s * 2^-14 * (m/4),          e == 0
    Infinity:  e == 31, m == 0
    NaN:       e == 31, m != 0

    Must round to nearest with ties-to-even (not truncate), and saturate to
    signed infinity only once the magnitude overflows the representable
    range under that same rounding rule. Must not modify `values`.
    """
    # BUG: this floors the mantissa instead of rounding to nearest, and
    # saturates to infinity immediately past the max finite value instead
    # of continuing to round-to-nearest across the overflow boundary.
    values = np.asarray(values, dtype=np.float32).astype(np.float64)
    shape = values.shape
    flat = values.ravel()
    sign_bit = np.signbit(flat).astype(np.uint8)
    av = np.abs(flat)

    out = np.zeros(flat.shape, dtype=np.uint8)

    nan_mask = np.isnan(flat)
    inf_mask = np.isinf(flat)
    overflow_mask = (~nan_mask) & (~inf_mask) & (av > 57344.0)
    zero_mask = (~nan_mask) & (~inf_mask) & (~overflow_mask) & (av == 0.0)
    finite_mask = (~nan_mask) & (~inf_mask) & (~overflow_mask) & (~zero_mask)

    if np.any(finite_mask):
        av_f = av[finite_mask]
        with np.errstate(divide="ignore"):
            e_unbiased = np.floor(np.log2(av_f)).astype(np.int64)
        e_stored = np.clip(e_unbiased + 15, 0, 30)

        is_sub = e_stored == 0
        scale = np.where(is_sub, 2.0 ** -14.0, np.exp2((e_stored - 15).astype(np.float64)))
        significand = np.where(is_sub, av_f / (scale / 4.0), (av_f / scale - 1.0) * 4.0)
        m = np.floor(significand).astype(np.int64)  # truncate, not round
        m = np.clip(m, 0, 3)

        code = (e_stored << 2) | m
        out[finite_mask] = code.astype(np.uint8)

    out[overflow_mask] = 0x7C
    out[inf_mask] = 0x7C
    out[nan_mask] = 0x7F

    out = (sign_bit << 7) | out
    return out.reshape(shape)
