import numpy as np


def per_axis_qint8(W: np.ndarray, axis: int = 0):
    """
    Symmetric int8 quantization with one scale per index along `axis`
    (per-output-channel when axis=0): scale = absmax / 127 (no
    zero-point), codes = clip(round(W / scale), -127, 127).
    Returns (codes, scale, dequant).
    """
    W = np.asarray(W, dtype=np.float64)
    reduce_axes = tuple(a for a in range(W.ndim) if a != axis)
    absmax = np.max(np.abs(W), axis=reduce_axes, keepdims=True)
    absmax = np.where(absmax == 0.0, 1.0, absmax)
    scale = absmax / 127.0
    codes = np.clip(np.round(W / scale), -127, 127)
    deq = codes * scale
    return codes, scale, deq
