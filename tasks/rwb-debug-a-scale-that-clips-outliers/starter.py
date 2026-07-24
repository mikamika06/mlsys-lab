import numpy as np


def quantize_absmax(x: np.ndarray) -> tuple[np.ndarray, float]:
    # TODO: fix the scale computation. This version uses 3*std, which clips
    # rare large values instead of preserving the full tensor range.
    x = np.asarray(x, dtype=np.float32)
    scale = float(3.0 * np.std(x) / 127.0)
    if scale == 0.0:
        scale = 1.0
    q = np.clip(np.round(x / scale), -127, 127).astype(np.int8)
    return q, scale
