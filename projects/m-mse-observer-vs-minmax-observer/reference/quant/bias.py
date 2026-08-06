import numpy as np


def compute_zero_point_bias(x: np.ndarray, scale: float, zero_point: int) -> float:
    q = np.clip(np.round(x / scale) + zero_point, 0, 255)
    dequant = (q - zero_point) * scale
    biased = np.mean(x - dequant)
    return float(biased)
