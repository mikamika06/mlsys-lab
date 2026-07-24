import numpy as np


def fp8_format_errors(x: np.ndarray) -> tuple[float, float]:
    """
    Quantize-then-dequantize `x` through the two OCP 8-bit float formats,
    E4M3 and E5M2, with NO rescaling (raw values, saturating at each
    format's finite max magnitude). Return the maximum absolute
    reconstruction error for each format as (e4m3_max_abs_err, e5m2_max_abs_err).
    """
    raise NotImplementedError('your code here')
