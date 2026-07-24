import numpy as np


def rsqrt_raw(x: np.ndarray) -> np.ndarray:
    """Magic-constant (0x5f3759df) bit-hack approximation of 1/sqrt(x).

    Reinterpret ``x`` as uint32 bits, compute
    ``0x5f3759df - (bits >> 1)``, and reinterpret the result back as
    float32. No refinement step here.
    """
    raise NotImplementedError('your code here')


def rsqrt_newton(x: np.ndarray) -> np.ndarray:
    """rsqrt_raw(x) refined by exactly one Newton-Raphson step.

    y1 = y0 * (1.5 - 0.5 * x * y0**2), where y0 = rsqrt_raw(x).
    """
    raise NotImplementedError('your code here')
