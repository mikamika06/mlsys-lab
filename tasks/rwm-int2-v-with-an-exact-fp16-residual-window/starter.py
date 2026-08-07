import math

def kv_int2_residual_window(V: list[list[float]], group_size: int=32, residual_window: int=16) -> list[list[float]]:
    """
    Quantize all but the last `residual_window` rows of `V` to 2 bits/element
    using grouped affine (zero-point) quantization along the channel axis;
    leave the last `residual_window` rows exact. Returns the reconstructed
    (T, d) array.
    """
    raise NotImplementedError('your code here')
