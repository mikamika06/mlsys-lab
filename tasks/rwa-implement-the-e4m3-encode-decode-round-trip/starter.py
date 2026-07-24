import numpy as np


def e4m3_round_trip(x: np.ndarray) -> np.ndarray:
    """Simulate round-tripping `x` through E4M3FN: encode to the 8-bit
    format then decode straight back to float.

    E4M3FN: 1 sign bit, 4 exponent bits (bias 7), 3 mantissa bits.
      - normal:    (-1)^s * (1 + m/8) * 2^(e-7),   e in [1, 14]
      - subnormal: (-1)^s * (m/8) * 2^-6,          e == 0
      - largest finite magnitude: 448 (e=14, m=7); anything larger
        saturates to +-448.
      - no infinities; the code e=15, m=7 is reserved (NaN), never produced.

    Round the input to the nearest representable E4M3FN value using
    round-to-nearest-even (on an exact tie, pick the candidate whose
    stored mantissa has an even low bit), then decode back.

    Parameters
    ----------
    x : ndarray of float
        Values to round-trip.

    Returns
    -------
    ndarray, float32, same shape as `x`.
    """
    raise NotImplementedError('your code here')
