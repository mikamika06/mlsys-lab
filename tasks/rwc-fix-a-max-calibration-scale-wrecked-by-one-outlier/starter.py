import numpy as np


def calibrate_scale_and_dequantize(x, qmax=127, percentile=99.0):
    # TODO: replace max calibration with percentile calibration.
    # This implementation is broken because one outlier controls the scale.
    x = np.asarray(x, dtype=np.float64)
    amax = float(np.max(np.abs(x)))
    scale = amax / qmax
    q = np.clip(np.round(x / scale), -qmax, qmax)
    reconstructed = q * scale
    return amax, scale, reconstructed
