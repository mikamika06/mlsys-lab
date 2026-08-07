import numpy as np


def quantize_q4_0(tensor: np.ndarray) -> bytes:
    raise NotImplementedError


def dequantize_q4_0(data: bytes, shape: tuple) -> np.ndarray:
    raise NotImplementedError


def max_abs_err(original: np.ndarray, reconstructed: np.ndarray) -> float:
    raise NotImplementedError
