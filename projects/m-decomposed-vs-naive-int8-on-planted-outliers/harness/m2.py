import ref
import numpy as np


def check(workdir):
    from quant.threshold import find_optimal_threshold
    x, w = ref.generate_data()
    t = find_optimal_threshold(x, w, max_fp16_flops=5000.0)
    out = {"threshold_optimal": 1.0 if isinstance(t, (float, int, np.number)) and t > 0 else 0.0}
    if out["threshold_optimal"] == 0.0:
        out["_note"] = f"invalid threshold returned: {t}"
    return out
