import ref
import numpy as np


def check(workdir):
    from palettize.scale import per_channel_scale
    out = {"mse_score": 1.0}
    try:
        q, s, z = per_channel_scale(ref.TEST_WEIGHT, 4)
        dec = (q.astype(np.float32) - z) * s
        mse = float(np.mean((ref.TEST_WEIGHT - dec) ** 2))
        out["mse_score"] = mse
    except Exception as e:
        out["_note"] = f"Error: {e}"
        out["mse_score"] = 999.0
    return out
