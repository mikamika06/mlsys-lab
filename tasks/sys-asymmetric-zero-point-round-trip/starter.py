import numpy as np


def affine_quant_dequant(x: np.ndarray, qmin: int, qmax: int) -> np.ndarray:
    """
    Asymmetric affine (zero-point) quantize-then-dequantize round trip:
    scale = (max(0,x.max()) - min(0,x.min())) / (qmax - qmin), zero_point
    from scale, codes = clip(round(x/scale + zp), qmin, qmax), dequant =
    (codes - zp) * scale. See task.md for the exact formulas.
    """
    raise NotImplementedError('your code here')
