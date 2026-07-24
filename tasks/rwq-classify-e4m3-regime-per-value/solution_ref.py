import numpy as np

# E4M3(FN) format: 1 sign bit, 4 exponent bits (bias 7), 3 mantissa bits.
MIN_SUBNORMAL = 2.0 ** -9   # smallest representable nonzero magnitude
MIN_NORMAL = 2.0 ** -6      # smallest magnitude with a normal (implicit-1) exponent
MAX_NORMAL = 448.0          # largest finite representable magnitude

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
    a = np.abs(x)

    out = np.full(x.shape, OVERFLOW_CLAMPED, dtype=np.int64)
    out = np.where(a <= MAX_NORMAL, NORMAL, out)
    out = np.where(a < MIN_NORMAL, SUBNORMAL, out)
    out = np.where(a < MIN_SUBNORMAL, UNDERFLOW_TO_ZERO, out)
    # NaN: `a <= MAX_NORMAL` is False for NaN, so it falls through to
    # OVERFLOW_CLAMPED, which is the desired "can't be represented" bucket.
    return out.astype(np.int64)
