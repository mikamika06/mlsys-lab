import numpy as np


def quantize_q4_k(weights: np.ndarray) -> bytes:
    raise NotImplementedError


def dequantize_q4_k(data: bytes) -> np.ndarray:
    raise NotImplementedError
