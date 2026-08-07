import numpy as np

def generate_fixture():
    np.random.seed(42)
    # Generate weights that are strictly positive,
    # to highlight how zeros distort asymmetric min/max.
    return np.random.uniform(5.0, 15.0, 200)

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

def joint_recipe(w: np.ndarray, p: float, b: int) -> np.ndarray:
    w_out = w.copy()
    n = int(np.round(w.size * p))
    mask = np.ones(w.size, dtype=bool)
    if n > 0:
        idx = np.argsort(np.abs(w))[:n]
        mask[idx] = False
        w_out[idx] = 0.0

    active = w[mask]
    if len(active) == 0:
        return w_out

    w_min, w_max = np.min(active), np.max(active)
    if w_min != w_max:
        levels = (1 << b) - 1
        scale = (w_max - w_min) / levels
        zp = np.round(-w_min / scale)
        q = np.round(active / scale) + zp
        q = np.clip(q, 0, levels)
        w_out[mask] = (q - zp) * scale

    return w_out
