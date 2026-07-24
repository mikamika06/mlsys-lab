import numpy as np


def fp64_to_fp32_bits(x: np.ndarray) -> np.ndarray:
    """Cast float64 -> float32 by hand, using guard/round/sticky
    round-to-nearest-even on the bit fields (sign, exponent, mantissa).

    Return a uint32 array of float32 bit patterns, same shape as x.
    """
    raise NotImplementedError('your code here')
