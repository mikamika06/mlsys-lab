import numpy as np


def compare_mxfp4_nvfp4(weights: np.ndarray) -> np.ndarray:
    """
    weights: array of any shape.

    Quantize-then-dequantize `weights` with both 4-bit microscaling
    schemes (both use the E2M1 FP4 grid [0, 0.5, 1, 1.5, 2, 3, 4, 6] per
    element, only the shared block scale differs):

    - MXFP4: block of 32 elements, shared scale restricted to a power of
      two (an E8M0-style exponent-only scale).
    - NVFP4: block of 16 elements, shared scale rounded to the nearest
      real FP8 E4M3 value (not restricted to a power of two).

    Returns np.array([mxfp4_rel_err, nvfp4_rel_err]), the global relative
    L2 reconstruction error for each scheme against the original weights.
    """
    raise NotImplementedError('your code here')
