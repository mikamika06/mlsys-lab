import numpy as np


def dequantize_linear(q: np.ndarray, scale, zero_point, axis: int = 0) -> np.ndarray:
    """QDQ-style DequantizeLinear: deq = (q - zero_point) * scale.

    q: integer code array of any shape.
    scale, zero_point: either scalars (per-tensor) or 1-D arrays of length
      q.shape[axis] (per-axis), broadcast against `q` along `axis`.

    Returns a float64 array, same shape as `q`.
    """
    raise NotImplementedError('your code here')
