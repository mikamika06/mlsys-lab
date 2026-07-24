"""Reference solution for `num-nextafter-x-inf-by-integer-bit-increment`."""
from __future__ import annotations

import numpy as np


def next_up(x: np.ndarray) -> np.ndarray:
    """Elementwise IEEE-754 nextafter(x, +inf) for finite float32 `x`,
    computed purely via integer manipulation of the raw sign-magnitude
    bit pattern (no `np.nextafter` call).

    float32 bits are `[sign:1][exponent:8][mantissa:23]`, i.e.
    sign-magnitude, not two's complement: the 31 low bits, read as an
    unsigned integer, are already monotonically increasing with |x|.

    * Non-negative x (sign bit 0): moving toward +inf means moving away
      from zero, i.e. incrementing the magnitude by 1 ULP.
    * Negative x (sign bit 1): moving toward +inf means moving *toward*
      zero, i.e. DECREMENTING the magnitude by 1 ULP.
    * Special case: -0.0 has magnitude 0, so "decrementing" is undefined;
      by IEEE-754, nextafter(-0, +inf) == nextafter(+0, +inf) == the
      smallest positive subnormal.
    """
    x = np.asarray(x, dtype=np.float32)
    bits = x.view(np.uint32).copy()

    sign = bits >> np.uint32(31)
    mag = bits & np.uint32(0x7FFFFFFF)

    pos_mask = sign == 0
    neg_zero_mask = (sign == 1) & (mag == 0)
    neg_other_mask = (sign == 1) & (mag != 0)

    out = np.empty_like(bits)
    out[pos_mask] = mag[pos_mask] + np.uint32(1)
    out[neg_zero_mask] = np.uint32(1)
    out[neg_other_mask] = (np.uint32(1) << np.uint32(31)) | (mag[neg_other_mask] - np.uint32(1))

    return out.view(np.float32)
