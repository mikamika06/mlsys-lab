import numpy as np


def check(workdir):
    from quant.metrics import compute_size_ratio, compute_quality_delta
    out = {"ratio_match": 0.0, "delta_match": 0.0}

    orig_b = 1000
    quant_b = 250
    ratio = compute_size_ratio(orig_b, quant_b)
    if abs(ratio - 0.25) < 1e-5:
        out["ratio_match"] = 1.0

    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([1.1, 1.9, 3.2], dtype=np.float32)
    delta = compute_quality_delta(x, y)
    if abs(delta - 0.2) < 1e-5:
        out["delta_match"] = 1.0

    return out
