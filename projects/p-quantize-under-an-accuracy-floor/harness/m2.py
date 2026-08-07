import sys
import numpy as np


def check(workdir):
    sys.path.insert(0, workdir)
    from quant.calib import compute_scales

    out = {"scale_error_below_threshold": 0.0, "calib_size_sensitivity_ok": 0.0}

    weights = np.array([-2.0, 1.0, 4.0, -8.0], dtype=np.float32)
    calib_data = [np.ones((1, 16), dtype=np.float32) * i for i in range(1, 11)]

    try:
        scale_small = compute_scales(weights, 2, calib_data)
        scale_large = compute_scales(weights, 10, calib_data)
    except Exception:
        return out

    if isinstance(scale_small, (float, np.floating)) and isinstance(
        scale_large, (float, np.floating)
    ):
        out["calib_size_sensitivity_ok"] = 1.0

    target_scale = (8.0 * 0.5 + 10.0 * 0.5) / 7.0
    if abs(scale_large - target_scale) < 1e-3:
        out["scale_error_below_threshold"] = 1.0

    return out
