import numpy as np


def affine_quant_dequant(x: np.ndarray, qmin: int, qmax: int) -> np.ndarray:
    """
    Standard asymmetric affine (zero-point) quantize/dequantize
    round-trip. The representable range is forced to include 0
    (min(0, x.min()), max(0, x.max())) so the zero-point never needs to
    be clamped away from its natural value.
    """
    x = np.asarray(x, dtype=np.float64)
    mn = min(0.0, float(np.min(x)))
    mx = max(0.0, float(np.max(x)))
    scale = (mx - mn) / (qmax - qmin) if mx > mn else 1.0
    zp = int(np.clip(round(qmin - mn / scale), qmin, qmax))

    codes = np.clip(np.round(x / scale + zp), qmin, qmax)
    return (codes - zp) * scale
