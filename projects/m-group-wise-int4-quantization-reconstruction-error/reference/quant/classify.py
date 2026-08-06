import numpy as np


def classify_saturation_elements(tensor: np.ndarray, scale: float, zero_point: int, qmin: int = -8, qmax: int = 7):
    quantized = np.round(tensor / scale) + zero_point
    clipped = (quantized < qmin) | (quantized > qmax)
    return clipped.astype(np.int32)
