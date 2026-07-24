import numpy as np


def im2col_patches(x: np.ndarray, kh: int, kw: int, stride: int) -> np.ndarray:
    """Zero-copy im2col patch extraction.

    Return a view of shape (out_h, out_w, kh, kw) over x's own memory
    buffer, built with explicit strides (e.g. numpy.lib.stride_tricks.
    as_strided) — no copying.
    """
    raise NotImplementedError('your code here')
