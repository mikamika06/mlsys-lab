import numpy as np


def fp64_to_fp32_bits(x: np.ndarray) -> np.ndarray:
    """Cast float64 -> float32 by hand, using GRS round-to-nearest-even.

    Returns the resulting float32 bit pattern as uint32, matching
    ``x.astype(np.float32).view(np.uint32)`` exactly for inputs that stay
    within float32's normal exponent range.
    """
    x = np.asarray(x, dtype=np.float64)
    bits64 = x.view(np.uint64)

    sign = (bits64 >> np.uint64(63)) & np.uint64(1)
    exp64 = ((bits64 >> np.uint64(52)) & np.uint64(0x7FF)).astype(np.int64)
    mant64 = bits64 & np.uint64(0xFFFFFFFFFFFFF)  # low 52 bits

    # rebias: float64 bias 1023 -> float32 bias 127
    exp32 = exp64 - 1023 + 127

    # split the 52-bit mantissa into kept (top 23) and dropped (low 29, GRS)
    kept = mant64 >> np.uint64(29)
    dropped = mant64 & np.uint64((1 << 29) - 1)
    half = np.uint64(1 << 28)

    round_up = (dropped > half) | (
        (dropped == half) & ((kept & np.uint64(1)) == np.uint64(1))
    )

    kept_rounded = kept + round_up.astype(np.uint64)
    carry = ((kept_rounded >> np.uint64(23)) & np.uint64(1)).astype(np.int64)
    kept_final = np.where(carry.astype(bool), np.uint64(0), kept_rounded)
    exp32_final = exp32 + carry

    sign32 = sign.astype(np.uint32)
    exp32_u = exp32_final.astype(np.uint32) & np.uint32(0xFF)
    mant32 = kept_final.astype(np.uint32) & np.uint32(0x7FFFFF)

    bits32 = (sign32 << np.uint32(31)) | (exp32_u << np.uint32(23)) | mant32
    return bits32.astype(np.uint32)
