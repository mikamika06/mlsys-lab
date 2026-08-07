import numpy as np


def decode_weight(quantized: np.ndarray, scale: np.ndarray, zero_point: np.ndarray):
    return (quantized.astype(np.float32) - zero_point) * scale
