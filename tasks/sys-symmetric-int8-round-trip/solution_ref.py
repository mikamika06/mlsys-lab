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
    w = np.asarray(w, dtype=np.float32)
    m = float(np.max(np.abs(w))) if w.size else 0.0
    scale = m / 127.0 if m > 0.0 else 1.0
    codes = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
    return codes, scale


def dequantize_symmetric_int8(codes: np.ndarray, scale: float) -> np.ndarray:
    """Invert `quantize_symmetric_int8`: codes.astype(float32) * scale."""
    codes = np.asarray(codes, dtype=np.int8)
    return codes.astype(np.float32) * np.float32(scale)
