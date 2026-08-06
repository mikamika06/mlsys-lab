import numpy as np


def fp64_to_fp32_bits(x: np.ndarray) -> np.ndarray:
    """Cast float64 -> float32 by hand, using GRS round-to-nearest-even."""
    arr = np.asarray(x, dtype=np.float64)
    bits64_arr = arr.view(np.uint64)
    out = np.empty(arr.shape, dtype=np.uint32)

    bits_flat = bits64_arr.reshape(-1)
    out_flat = out.reshape(-1)

    for i in range(len(bits_flat)):
        val = int(bits_flat[i])

        sign = (val >> 63) & 1
        exp64 = (val >> 52) & 0x7FF
        mant64 = val & 0xFFFFFFFFFFFFF

        exp32 = exp64 - 1023 + 127

        kept = mant64 >> 29
        dropped = mant64 & 0x1FFFFFFF
        half = 1 << 28

        if dropped > half:
            round_up = 1
        elif dropped == half:
            round_up = kept & 1
        else:
            round_up = 0

        kept_rounded = kept + round_up
        carry = (kept_rounded >> 23) & 1

        if carry != 0:
            kept_final = 0
        else:
            kept_final = kept_rounded

        exp32_final = exp32 + carry

        sign32 = sign
        exp32_u = exp32_final & 0xFF
        mant32 = kept_final & 0x7FFFFF

        bits32 = (sign32 << 31) | (exp32_u << 23) | mant32
        out_flat[i] = bits32

    return out
