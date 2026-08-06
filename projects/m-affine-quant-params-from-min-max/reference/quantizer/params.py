import numpy as np


def calc_affine_params(val_min: float, val_max: float, qmin: int = 0, qmax: int = 255) -> tuple[float, int]:
    r_min = min(val_min, 0.0)
    r_max = max(val_max, 0.0)
    if r_min == r_max:
        return 1.0, qmin
    scale = (r_max - r_min) / float(qmax - qmin)
    initial_zp = qmin - (r_min / scale)
    zp = int(np.round(initial_zp))
    zp_clamped = max(qmin, min(qmax, zp))
    return float(scale), zp_clamped


def quantize(x: np.ndarray, scale: float, zero_point: int, qmin: int = 0, qmax: int = 255) -> np.ndarray:
    q = np.round(x / scale) + zero_point
    q_clipped = np.clip(q, qmin, qmax)
    dtype = np.uint8 if qmin >= 0 else np.int8
    return q_clipped.astype(dtype)


def dequantize(q: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    return (q.astype(np.float32) - zero_point) * scale
