import numpy as np

def prune(w: np.ndarray, p: float) -> np.ndarray:
    w_out = w.copy()
    n = int(np.round(w.size * p))
    if n > 0:
        idx = np.argsort(np.abs(w))[:n]
        w_out[idx] = 0.0
    return w_out

def quantize(w: np.ndarray, b: int) -> np.ndarray:
    w_min, w_max = np.min(w), np.max(w)
    if w_min == w_max:
        return w.copy()
    levels = (1 << b) - 1
    scale = (w_max - w_min) / levels
    zp = np.round(-w_min / scale)
    q = np.round(w / scale) + zp
    q = np.clip(q, 0, levels)
    return (q - zp) * scale
