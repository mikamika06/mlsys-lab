import numpy as np

def calibration_params(tensor):
    """Compute uint8 asymmetric calibration parameters from observed min/max.

    Returns (scale, zero_point) where:
      scale     = (max_val - min_val) / 255
      zero_point = clamp(round(-min_val / scale), 0, 255)
    If the tensor is constant, returns (0.0, 0).
    """
    raise NotImplementedError
