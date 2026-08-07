import numpy as np


def quantize_q4_k(x: np.ndarray) -> bytes:
    raise NotImplementedError


def dequantize_q4_k(b: bytes, shape: tuple) -> np.ndarray:
    raise NotImplementedError


def quantize_q4_0(x: np.ndarray) -> bytes:
    raise NotImplementedError


def dequantize_q4_0(b: bytes, shape: tuple) -> np.ndarray:
    raise NotImplementedError
