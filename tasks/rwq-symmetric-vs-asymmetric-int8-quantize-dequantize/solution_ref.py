import numpy as np

def sym_quant_dequant(x: np.ndarray) -> np.ndarray:
    absmax = np.max(np.abs(x))
    scale = absmax / 127 if absmax != 0 else 1.0
    q = np.round(x / scale)
    q = np.clip(q, -128, 127).astype(np.int8)
    return q.astype(np.float64) * scale

def asym_quant_dequant(x: np.ndarray) -> np.ndarray:
    mn = x.min()
    mx = x.max()
    rng = mx - mn
    if rng == 0:
        scale = 1.0
        zp = 128
    else:
        scale = rng / 255
        zp = int(np.round(-mn / scale))
    q = np.round(x / scale + zp)
    q = np.clip(q, 0, 255).astype(np.uint8)
    return (q.astype(np.float64) - zp) * scale
