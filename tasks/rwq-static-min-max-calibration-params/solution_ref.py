import numpy as np

def calibration_params(tensor):
    """Compute uint8 asymmetric calibration parameters from observed min/max.

    Returns (scale, zero_point) where:
      scale     = (max_val - min_val) / 255
      zero_point = clamp(round(-min_val / scale), 0, 255)
    If the tensor is constant, returns (0.0, 0).
    """
    t = np.asarray(tensor, dtype=np.float64)
    mn = float(t.min())
    mx = float(t.max())
    scale = (mx - mn) / 255.0
    if scale == 0.0:
        return 0.0, 0
    zero_point = int(np.clip(np.round(-mn / scale), 0, 255))
    return scale, zero_point
