import numpy as np


def per_axis_qint8(W: np.ndarray, axis: int = 0):
    """
    Return (codes, scale, dequant) for symmetric int8 quantization of W
    with one scale per index along `axis`: scale = absmax/127 (absmax
    reduced over all other axes), codes = clip(round(W/scale), -127, 127),
    no zero-point. See task.md.
    """
    raise NotImplementedError('your code here')
