import numpy as np

MIN_SUBNORMAL = 2.0 ** -9
MIN_NORMAL = 2.0 ** -6
MAX_NORMAL = 448.0

UNDERFLOW_TO_ZERO = 0
SUBNORMAL = 1
NORMAL = 2
OVERFLOW_CLAMPED = 3


def classify_e4m3_regime(x: np.ndarray) -> np.ndarray:
    """
    Classify each element of `x` by which E4M3 representable regime its
    magnitude falls into:

      0 = underflow_to_zero : |x| < MIN_SUBNORMAL (rounds to 0; includes x == 0)
      1 = subnormal         : MIN_SUBNORMAL <= |x| < MIN_NORMAL
      2 = normal             : MIN_NORMAL <= |x| <= MAX_NORMAL
      3 = overflow_clamped  : |x| > MAX_NORMAL (including +/-inf, nan)

    Returns an int64 NumPy array of the same shape as `x`.
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.empty(x.shape, dtype=np.int64)
    for i, val in enumerate(x.flat):
        a = -val if val < 0 else val
        if a < MIN_SUBNORMAL:
            res = UNDERFLOW_TO_ZERO
        elif a < MIN_NORMAL:
            res = SUBNORMAL
        elif a <= MAX_NORMAL:
            res = NORMAL
        else:
            res = OVERFLOW_CLAMPED
        out.flat[i] = res
    return out
