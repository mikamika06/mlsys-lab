import numpy as np


def calibrate_scale_and_dequantize(x, qmax=127, percentile=99.0):
    x = np.asarray(x, dtype=np.float64)
    amax = float(np.percentile(np.abs(x), percentile))
    scale = amax / qmax
    q = np.clip(np.round(x / scale), -qmax, qmax)
    reconstructed = q * scale
    return amax, scale, reconstructed
