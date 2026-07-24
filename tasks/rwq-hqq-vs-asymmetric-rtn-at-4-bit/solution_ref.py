import numpy as np


def _rtn_reconstruct(x):
    qmax = 15.0
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    scale = (xmax - xmin) / qmax
    if scale == 0:
        return x.copy()
    zero = int(np.round(-xmin / scale))
    q = np.clip(np.round(x / scale) + zero, 0, 15)
    return scale * (q - zero)


def _hqq_reconstruct(x):
    qmax = 15.0
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    rtn_scale = (xmax - xmin) / qmax
    if rtn_scale == 0:
        return x.copy()

    best_obj = float("inf")
    best = None
    for scale in np.linspace(0.5 * rtn_scale, 1.5 * rtn_scale, 101):
        for zero in range(-32, 33):
            q = np.clip(np.round(x / scale) + zero, 0, 15)
            xhat = scale * (q - zero)
            obj = float(np.sum(np.abs(x - xhat) ** 0.7))
            if obj < best_obj:
                best_obj = obj
                best = xhat
    return best


def compare_4bit_quantizers(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    hqq = _hqq_reconstruct(x)
    rtn = _rtn_reconstruct(x)
    return float(np.mean((hqq - x) ** 2)), float(np.mean((rtn - x) ** 2))
