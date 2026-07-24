import numpy as np


def quantize_symmetric_int8(w: np.ndarray):
    """Symmetric per-tensor int8 quantization.

    scale = max(|w|) / 127   (or 1.0 if `w` is all zeros, to avoid a
    division by zero -- the codes are then all zero too, so the scale's
    value does not matter).

    codes = round(w / scale), clipped to [-127, 127] and cast to int8.

    Parameters
    ----------
    w : array_like, any shape.

    Returns
    -------
    codes : np.ndarray, dtype int8, same shape as `w`.
    scale : float
    """
    raise NotImplementedError('your code here')


def dequantize_symmetric_int8(codes: np.ndarray, scale: float) -> np.ndarray:
    """Invert `quantize_symmetric_int8`: codes.astype(float32) * scale."""
    raise NotImplementedError('your code here')
