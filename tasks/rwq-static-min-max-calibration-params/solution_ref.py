import numpy as np

def calibration_params(tensor):
    """Compute uint8 asymmetric calibration parameters from observed min/max.

    Returns (scale, zero_point) where:
      scale     = (max_val - min_val) / 255
      zero_point = clamp(round(-min_val / scale), 0, 255)
    If the tensor is constant, returns (0.0, 0).
    """
    t = np.asarray(tensor, dtype=np.float64)
    iterator = t.flat
    mn = float(next(iterator))
    mx = mn
    for val in iterator:
        if val < mn:
            mn = float(val)
        if val > mx:
            mx = float(val)
    scale = (mx - mn) / 255.0
    if scale == 0.0:
        return 0.0, 0
    val_rounded = round(-mn / scale)
    if val_rounded < 0:
        zero_point = 0
    elif val_rounded > 255:
        zero_point = 255
    else:
        zero_point = int(val_rounded)
    return scale, zero_point
