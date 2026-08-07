import numpy as np


def compute_scales(tensor, mode="per-tensor"):
    if mode == "per-tensor":
        amax = np.max(np.abs(tensor))
        scale = amax / 7.0 if amax > 0 else 1.0
        return np.array([scale], dtype=np.float32)
    elif mode == "per-head":
        amax = np.max(np.abs(tensor), axis=-1, keepdims=True)
        scales = amax / 7.0
        scales[scales == 0] = 1.0
        return scales.astype(np.float32)
    else:
        raise ValueError("unknown mode")
