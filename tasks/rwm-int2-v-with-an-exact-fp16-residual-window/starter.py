import numpy as np


def kv_int2_residual_window(V: np.ndarray, group_size: int = 32, residual_window: int = 16) -> np.ndarray:
    """
    Quantize all but the last `residual_window` rows of `V` to 2 bits/element
    using grouped affine (zero-point) quantization along the channel axis;
    leave the last `residual_window` rows exact. Returns the reconstructed
    (T, d) array.
    """
    raise NotImplementedError('your code here')
