import numpy as np


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
    raise NotImplementedError('your code here')
