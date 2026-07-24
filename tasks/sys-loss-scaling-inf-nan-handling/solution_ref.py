import numpy as np


def unscale_and_check(scaled_grads, scale):
    skip = any(
        not np.all(np.isfinite(np.asarray(g, dtype=np.float32))) for g in scaled_grads
    )
    unscaled = [
        np.asarray(g, dtype=np.float32) / np.float32(scale) for g in scaled_grads
    ]
    return bool(skip), unscaled
