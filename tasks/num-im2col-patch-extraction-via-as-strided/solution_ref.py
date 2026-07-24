import numpy as np
from numpy.lib.stride_tricks import as_strided


def im2col_patches(x: np.ndarray, kh: int, kw: int, stride: int) -> np.ndarray:
    """Zero-copy im2col patch extraction via as_strided.

    Returns a view of shape (out_h, out_w, kh, kw) over x's own buffer.
    """
    x = np.asarray(x, dtype=np.float64)
    H, W = x.shape
    out_h = (H - kh) // stride + 1
    out_w = (W - kw) // stride + 1
    s0, s1 = x.strides
    new_strides = (s0 * stride, s1 * stride, s0, s1)
    return as_strided(x, shape=(out_h, out_w, kh, kw), strides=new_strides)
