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
    m = 0.0
    for x in w.flat:
        abs_x = x if x >= 0.0 else -x
        if abs_x > m:
            m = abs_x
    m = float(m)
    scale = m / 127.0 if m > 0.0 else 1.0
    
    codes_list = []
    for x in w.flat:
        val = x / scale
        r = round(val)
        if r < -127:
            r = -127
        elif r > 127:
            r = 127
        codes_list.append(r)
    
    codes = np.array(codes_list, dtype=np.int8).reshape(w.shape)
    return codes, scale


def dequantize_symmetric_int8(codes: np.ndarray, scale: float) -> np.ndarray:
    """Invert `quantize_symmetric_int8`: codes.astype(float32) * scale."""
    codes = np.asarray(codes, dtype=np.int8)
    s = float(scale)
    res_list = []
    for c in codes.flat:
        res_list.append(float(c) * s)
    return np.array(res_list, dtype=np.float32).reshape(codes.shape)
