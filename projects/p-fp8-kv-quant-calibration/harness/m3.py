import numpy as np

def check(workdir):
    from kvquant.calib import calibrate_scale
    m = {"scale_calibrated": 0.0}

    np.random.seed(42)
    tensors = [np.random.randn(10, 32).astype(np.float32) * (i + 1) for i in range(5)]
    scale = calibrate_scale(tensors)

    expected_max = max(float(np.max(np.abs(t))) for t in tensors)
    if abs(scale - expected_max) < 1e-5:
        m["scale_calibrated"] = 1.0

    return m
